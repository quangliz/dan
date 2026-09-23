"""Qwen3.5 (hybrid Gated DeltaNet + gated attention), text only, prefill only.

Three of every four layers are Gated DeltaNet: a causal depthwise conv over
the q/k/v projections, then a gated delta rule, a linear recurrence with a
[heads, k_dim, v_dim] state. The rest are full attention with an output gate,
q/k RMSNorm and rotary on part of each head. Norms are zero-centered
(``x * (1 + w)``). Text-only multimodal RoPE is plain RoPE: all three position
streams are equal.

Tree reads through a recurrence: a segment's trunk runs from its initial state
(zeros, or the prefix cache's), and every branch then runs from the trunk's
final state and conv history, so questions see the prompt but not each other,
exactly as the attention layers' tree mask does. The prefix cache keeps, per
DeltaNet layer, the conv inputs of the last ``kernel - 1`` static tokens and
the recurrent state after them.

Numerics follow Hugging Face's reference implementation (Apache-2.0); on CUDA
the delta rule runs in flash-linear-attention's varlen kernel when installed.
"""
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn

from ..attention import attend

try:  # GPU kernels (pip install flash-linear-attention)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
except ImportError:  # pragma: no cover - CPU / not installed
    fla_chunk_gated_delta_rule = None


def text_config(config):
    return getattr(config, "text_config", None) or config


class RMSNorm(nn.Module):
    """Zero-centered: the weight is stored as w and applied as (1 + w)."""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = self.weight * y.to(dtype)
        return (y * F.silu(gate.float())).to(dtype)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def delta_rule_torch(q, k, v, g, beta, state, chunk=64):
    """Chunked gated delta rule for one sequence. q, k: [T, H, Dk]; v: [T, H, Dv];
    g, beta: [T, H]; state: [H, Dk, Dv] fp32. q and k are l2-normalized here.
    Returns ([T, H, Dv] in q's dtype, final state fp32)."""
    dtype, t = q.dtype, q.shape[0]
    q, k, v, beta, g = (x.transpose(0, 1).float() for x in (q, k, v, beta, g))  # heads first
    q, k = l2norm(q), l2norm(k)
    q = q * q.shape[-1] ** -0.5
    pad = (chunk - t % chunk) % chunk
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    h, n = q.shape[0], (t + pad) // chunk
    v_beta, k_beta = v * beta[..., None], k * beta[..., None]
    q, k, k_beta, v_beta = (x.reshape(h, n, chunk, x.shape[-1]) for x in (q, k, k_beta, v_beta))
    g = g.reshape(h, n, chunk).cumsum(-1)
    upper = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    decay = (g[..., :, None] - g[..., None, :]).masked_fill(upper, float("-inf")).exp()
    system = (k_beta @ k.transpose(-1, -2)) * decay
    intra = (q @ k.transpose(-1, -2)) * decay
    new_v = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
    k_cum = torch.linalg.solve_triangular(system, k_beta * g.exp()[..., None], upper=False, unitriangular=True)
    q = q * g.exp()[..., None]
    k = k * (g[..., -1:] - g).exp()[..., None]
    chunk_decay = g[..., -1].exp()[..., None, None]
    out = torch.zeros_like(new_v)
    for i in range(n):
        vi = new_v[:, i] - k_cum[:, i] @ state
        out[:, i] = q[:, i] @ state + intra[:, i] @ vi
        state = state * chunk_decay[:, i] + k[:, i].transpose(-1, -2) @ vi
    out = out.reshape(h, -1, out.shape[-1])[:, :t].transpose(0, 1)
    return out.to(dtype), state


def delta_rule(q, k, v, g, beta, lengths, states):
    """Gated delta rule over packed sequences of ``lengths`` (q etc. [N, H, D]),
    each starting from its ``states[i]`` ([S, H, Dk, Dv] fp32). Returns the
    packed output and the final states."""
    if fla_chunk_gated_delta_rule is not None and q.is_cuda:
        cu = torch.tensor([0, *lengths], device=q.device).cumsum(0)
        out, final = fla_chunk_gated_delta_rule(
            q[None], k[None], v[None], g[None], beta[None], initial_state=states.float(),
            output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
        return out[0], final
    outs, finals, i = [], [], 0
    for n, s in zip(lengths, states):
        o, f = delta_rule_torch(q[i:i + n], k[i:i + n], v[i:i + n], g[i:i + n], beta[i:i + n], s.float())
        outs.append(o)
        finals.append(f)
        i += n
    return torch.cat(outs), torch.stack(finals)


class DeltaNet(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.index = index
        self.hk, self.hv = c.linear_num_key_heads, c.linear_num_value_heads
        self.dk, self.dv = c.linear_key_head_dim, c.linear_value_head_dim
        self.kernel = c.linear_conv_kernel_dim
        key_dim, value_dim = self.hk * self.dk, self.hv * self.dv
        self.conv_dim = 2 * key_dim + value_dim
        self.in_proj_qkv = nn.Linear(c.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(c.hidden_size, value_dim, bias=False)
        self.in_proj_b = nn.Linear(c.hidden_size, self.hv, bias=False)
        self.in_proj_a = nn.Linear(c.hidden_size, self.hv, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.kernel, groups=self.conv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.hv))
        self.A_log = nn.Parameter(torch.zeros(self.hv))
        self.norm = GatedRMSNorm(self.dv, c.rms_norm_eps)
        self.out_proj = nn.Linear(value_dim, c.hidden_size, bias=False)

    def conv(self, x, runs):
        """Causal depthwise conv + SiLU over packed runs. ``runs``: (history
        [kernel-1, C], slice of x); each run's output sees only its history and
        itself. One conv1d over the runs laid end to end with their history."""
        ext, spans, at = [], [], 0
        for hist, sl in runs:
            ext += [hist, x[sl]]
            spans.append((at, sl.stop - sl.start))
            at += hist.shape[0] + sl.stop - sl.start
        y = F.conv1d(torch.cat(ext).T[None], self.conv1d.weight, groups=self.conv_dim)[0].T  # j reads ext[j:j+kernel]
        return F.silu(torch.cat([y[a:a + n] for a, n in spans]))

    def forward(self, x, segments):
        qkv, z = self.in_proj_qkv(x), self.in_proj_z(x)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias.float())
        k1, dtype = self.kernel - 1, x.dtype
        zeros_hist = qkv.new_zeros(k1, self.conv_dim)
        zeros_state = torch.zeros(self.hv, self.dk, self.dv, device=x.device, dtype=torch.float32)
        # Trunks: split at the cache boundary when a static prefix is saved.
        trunk_runs, trunk_states, trunk_of = [], [], []
        for si, s in enumerate(segments):
            hist, state = s.past[self.index] if s.past else (zeros_hist, zeros_state)
            cut = [s.start, s.start + s.save, s.start + s.trunk] if 0 < s.save < s.trunk else [s.start, s.start + s.trunk]
            for a, b in pairwise(cut):
                trunk_runs.append((hist, slice(a, b)))
                trunk_states.append(state)
                trunk_of.append(si)
                hist, state = None, None  # the later part continues from the earlier one: filled below
        out = torch.empty(x.shape[0], self.hv, self.dv, device=x.device, dtype=dtype)
        # Run trunk parts in order: a second part needs the first part's final state and conv tail.
        tails, finals = {}, {}
        for part in range(2):
            idx = [i for i, (h, _) in enumerate(trunk_runs) if (h is None) == bool(part)]
            if not idx:
                continue
            runs = []
            for i in idx:
                h, sl = trunk_runs[i]
                if h is None:
                    prev = i - 1
                    h = tails[prev]
                    trunk_states[i] = finals[prev]
                runs.append((h, sl))
            conv = self.conv(qkv, runs)
            lens = [sl.stop - sl.start for _, sl in runs]
            o, f = self._rule(conv, beta, g, runs, lens, torch.stack([trunk_states[i] for i in idx]))
            at = 0
            for j, (i, (h, sl), n) in enumerate(zip(idx, runs, lens)):
                out[sl] = o[at:at + n]
                at += n
                tails[i] = torch.cat([h, qkv[sl]])[-k1:]
                finals[i] = f[j]
        # Cache entries and each segment's trunk end state.
        seg_tail, seg_final = {}, {}
        for i, si in enumerate(trunk_of):
            s = segments[si]
            seg_tail[si], seg_final[si] = tails[i], finals[i]  # the last part wins
            if s.save and trunk_runs[i][1].stop == s.start + s.save:
                s.saved[self.index] = (tails[i].clone(), finals[i].clone())
        # Branches: every branch starts from its segment's trunk end.
        runs, states = [], []
        for si, s in enumerate(segments):
            for a, b in s.branches:
                runs.append((seg_tail[si], slice(a, b)))
                states.append(seg_final[si])
        if runs:
            conv = self.conv(qkv, runs)
            lens = [sl.stop - sl.start for _, sl in runs]
            o, _ = self._rule(conv, beta, g, runs, lens, torch.stack(states))
            at = 0
            for (_, sl), n in zip(runs, lens):
                out[sl] = o[at:at + n]
                at += n
        y = self.norm(out.reshape(-1, self.dv), z.reshape(-1, self.dv)).reshape(x.shape[0], -1)
        return self.out_proj(y)

    def _rule(self, conv, beta, g, runs, lens, states):
        key_dim = self.hk * self.dk
        q, k, v = conv.split([key_dim, key_dim, self.hv * self.dv], dim=-1)
        n = conv.shape[0]
        q, k, v = q.reshape(n, self.hk, self.dk), k.reshape(n, self.hk, self.dk), v.reshape(n, self.hv, self.dv)
        if self.hv > self.hk:
            q, k = q.repeat_interleave(self.hv // self.hk, 1), k.repeat_interleave(self.hv // self.hk, 1)
        rows = torch.cat([torch.arange(sl.start, sl.stop, device=conv.device) for _, sl in runs])
        return delta_rule(q, k, v, g[rows], beta[rows], lens, states)


class GatedAttention(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.index = index
        self.h, self.hkv, self.d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, self.h * self.d * 2, bias=c.attention_bias)
        self.k_proj = nn.Linear(c.hidden_size, self.hkv * self.d, bias=c.attention_bias)
        self.v_proj = nn.Linear(c.hidden_size, self.hkv * self.d, bias=c.attention_bias)
        self.o_proj = nn.Linear(self.h * self.d, c.hidden_size, bias=c.attention_bias)
        self.q_norm = RMSNorm(self.d, c.rms_norm_eps)
        self.k_norm = RMSNorm(self.d, c.rms_norm_eps)

    def forward(self, x, cos, sin, segments):
        n = x.shape[0]
        q, gate = self.q_proj(x).view(n, self.h, 2 * self.d).chunk(2, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(x).view(n, self.hkv, self.d))
        v = self.v_proj(x).view(n, self.hkv, self.d)
        q, k = rotate_partial(q, cos, sin), rotate_partial(k, cos, sin)
        o = attend(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), segments, self.index)
        o = o.transpose(0, 1).reshape(n, self.h * self.d) * torch.sigmoid(gate.reshape(n, -1))
        return self.o_proj(o)


def rotate_partial(x, cos, sin):
    """RoPE on the first ``cos.shape[-1]`` dims of each head (rotate-half style)."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    a, b = xr.chunk(2, dim=-1)
    return torch.cat([xr * cos[:, None] + torch.cat((-b, a), dim=-1) * sin[:, None], xp], dim=-1)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.kind = c.layer_types[index]
        if self.kind == "linear_attention":
            self.linear_attn = DeltaNet(c, index)
        else:
            self.self_attn = GatedAttention(c, index)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    def forward(self, x, cos, sin, segments):
        h = self.input_layernorm(x)
        h = self.linear_attn(h, segments) if self.kind == "linear_attention" else self.self_attn(h, cos, sin, segments)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class Body(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.num_hidden_layers))
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)


class Qwen35ForReads(nn.Module):
    MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})

    def __init__(self, config):
        super().__init__()
        c = text_config(config)
        self.config = c
        self.model = Body(c)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        if getattr(c, "tie_word_embeddings", False) or getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.register_buffer("inv_freq", self.frequencies(c), persistent=False)

    @staticmethod
    def frequencies(c):
        rp = c.rope_parameters
        dim = int(c.head_dim * rp.get("partial_rotary_factor", 1.0))
        return 1.0 / (rp["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))

    @staticmethod
    def remap(name):
        """Checkpoint key -> this module's key, or None to skip (vision, MTP)."""
        if name.startswith("model.language_model."):
            return "model." + name[len("model.language_model."):]
        if name.startswith("lm_head."):
            return name
        return None

    def hidden(self, ids, positions, segments):
        ang = positions.float()[:, None] * self.inv_freq[None, :]
        ang = torch.cat((ang, ang), dim=-1)
        x = self.model.embed_tokens(ids)
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        for layer in self.model.layers:
            x = layer(x, cos, sin, segments)
        return self.model.norm(x)

    def label_logits(self, h, label_ids):
        return self.lm_head.weight[label_ids] @ h
