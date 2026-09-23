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
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..attention import Batch, attend, cu_seqlens, repeat_heads, to_device
from ..fused import delta_gates, gated_rms_norm, rms_norm_zero_centered, rotate_partial, sigmoid_gate, swiglu
from .common import merge_linears

try:  # GPU kernels (pip install flash-linear-attention)
    import fla.ops.gated_delta_rule.chunk as fla_chunk
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fla_recurrent_gated_delta_rule
except ImportError:  # pragma: no cover - CPU / not installed
    fla_chunk = fla_chunk_gated_delta_rule = fla_recurrent_gated_delta_rule = None

# Calls with at most this many tokens in total use fla's fused recurrent kernel:
# one launch, token-sequential, ~0.15 ms to call. Bigger ones use the chunked
# kernel, parallel over tokens but ~1.2 ms of host time per call. On an L4 at
# Qwen3.5-4B shapes their wall times cross around 1-2k tokens, but in a
# GPU-bound forward the chunked kernel's host time is hidden and the recurrent
# kernel's token-sequential GPU time is not, hence the lower default.
RECURRENT_MAX_TOKENS = int(os.environ.get("DAN_RECURRENT_MAX_TOKENS", "512"))


def _memoize_fla_chunk_indices():
    """fla rebuilds varlen chunk indices on the host and uploads them on every
    call (a stream sync), caching only the latest call's arguments; trunk and
    branch calls alternate in every layer, so it always misses. Memoize by
    the host-side lengths instead: at most one upload per distinct batch shape."""
    original = fla_chunk.prepare_chunk_indices
    memo = {}

    def prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=None):
        if cu_seqlens_cpu is None:
            return original(cu_seqlens, chunk_size)
        key = (tuple(cu_seqlens_cpu.tolist()), chunk_size, cu_seqlens.device)
        hit = memo.get(key)
        if hit is None:
            if len(memo) > 4096:
                memo.clear()
            hit = memo[key] = original(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
        return hit

    fla_chunk.prepare_chunk_indices = prepare_chunk_indices


if fla_chunk is not None:
    _memoize_fla_chunk_indices()


def text_config(config):
    return getattr(config, "text_config", None) or config


class RMSNorm(nn.Module):
    """Zero-centered: the weight is stored as w and applied as (1 + w)."""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return rms_norm_zero_centered(x, self.weight, self.eps)


class GatedRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        return gated_rms_norm(x, gate, self.weight, self.eps)


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


def delta_rule(q, k, v, g, beta, lengths, states, cu=None):
    """Gated delta rule over packed sequences of ``lengths`` (q etc. [N, H, D]),
    each starting from its ``states[i]`` ([S, H, Dk, Dv] fp32). Returns the
    packed output and the final states. ``cu``: precomputed (device, host)
    cumulative lengths."""
    if fla_chunk_gated_delta_rule is not None and q.is_cuda:
        cu, cu_host = cu or cu_seqlens(lengths, q.device)
        if sum(lengths) <= RECURRENT_MAX_TOKENS:
            out, final = fla_recurrent_gated_delta_rule(
                q[None], k[None], v[None], g=g[None], beta=beta[None], initial_state=states.float(),
                output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
            return out[0], final
        out, final = fla_chunk_gated_delta_rule(
            q[None], k[None], v[None], g[None], beta[None], initial_state=states.float(),
            output_final_state=True, cu_seqlens=cu, cu_seqlens_cpu=cu_host, use_qk_l2norm_in_kernel=True)
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

    def merge(self):
        self.in_split = merge_linears(self, ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"], "in_proj")

    def forward(self, x, segments):
        if hasattr(self, "in_proj"):
            qkv, z, b, a = self.in_proj(x).split(self.in_split, dim=-1)
        else:
            qkv, z, b, a = self.in_proj_qkv(x), self.in_proj_z(x), self.in_proj_b(x), self.in_proj_a(x)
        beta, g = delta_gates(b, a, self.A_log, self.dt_bias)
        k1 = self.kernel - 1
        lay = getattr(segments, "recurrent", None)
        if lay is None or lay.rows != x.shape[0]:
            lay = RecurrentLayout.build(segments, k1, x.shape[0], x.device)
            if isinstance(segments, Batch):  # every DeltaNet layer of this forward shares it
                segments.recurrent = lay
        # Gather source: the packed rows, one zero row, then each cached conv history.
        extra = [qkv.new_zeros(1, self.conv_dim)] + [segments[i].past[self.index][0] for i in lay.past_segments]
        src = torch.cat([qkv, *extra])
        zeros = torch.zeros(self.hv, self.dk, self.dv, device=x.device, dtype=torch.float32)
        init = torch.stack([s.past[self.index][1].float() if s.past else zeros for s in segments])
        out = torch.empty(x.shape[0], self.hv, self.dv, device=x.device, dtype=x.dtype)
        final = self._phase(lay.trunk1, src, beta, g, init, out)
        if lay.trunk2 is not None:
            f2 = self._phase(lay.trunk2, src, beta, g, final.index_select(0, lay.split_segments), out)
            final_end = final.index_copy(0, lay.split_segments, f2)
        else:
            final_end = final
        for i, tail in lay.saves:  # prefix-cache entries at each save boundary
            segments[i].saved[self.index] = (src.index_select(0, tail), final[i].clone())
        if lay.branches is not None:
            self._phase(lay.branches, src, beta, g, final_end.index_select(0, lay.branch_segment), out)
        y = self.norm(out.reshape(-1, self.dv), z.reshape(-1, self.dv)).reshape(x.shape[0], -1)
        return self.out_proj(y)

    def _phase(self, ph, src, beta, g, states, out):
        """One batch of runs: causal conv over [history; run] (a gather), then
        the delta rule from ``states``; writes the runs' rows of ``out`` and
        returns their final states."""
        ext = src.index_select(0, ph.ext)
        y = F.conv1d(ext.T[None], self.conv1d.weight, groups=self.conv_dim)[0].T  # row j reads ext[j:j+kernel]
        conv = F.silu(y.index_select(0, ph.take))
        key_dim = self.hk * self.dk
        q, k, v = conv.split([key_dim, key_dim, self.hv * self.dv], dim=-1)
        n = conv.shape[0]
        q, k, v = q.reshape(n, self.hk, self.dk), k.reshape(n, self.hk, self.dk), v.reshape(n, self.hv, self.dv)
        q, k = repeat_heads(q, self.hv // self.hk), repeat_heads(k, self.hv // self.hk)
        o, f = delta_rule(q, k, v, g.index_select(0, ph.rows), beta.index_select(0, ph.rows), ph.lens, states, ph.cu)
        out.index_copy_(0, ph.rows, o)
        return f


@dataclass
class Phase:
    ext: torch.Tensor  # gather rows (from the conv source) of every run's [history; tokens]
    take: torch.Tensor  # conv output rows that belong to run tokens
    rows: torch.Tensor  # packed rows of the run tokens, in run order
    lens: list
    cu: tuple  # cumulative run lengths (device, host)


@dataclass
class RecurrentLayout:
    """Gather indices for the recurrent layers, built once per forward.

    Each segment has a virtual sequence: ``kernel - 1`` history rows (zeros,
    or its cached conv history) followed by its trunk rows. Runs: trunk part 1
    (up to the prefix-cache boundary, when one is being saved), trunk part 2
    (the rest), and every branch (whose history is the trunk's last rows)."""

    rows: int
    trunk1: Phase
    trunk2: Phase | None
    branches: Phase | None
    split_segments: torch.Tensor | None  # segments whose trunk is in two parts
    branch_segment: torch.Tensor | None  # segment of each branch
    past_segments: list  # segments with a cached conv history, in source order
    saves: list  # (segment, gather rows of its conv tail at the save boundary)

    @classmethod
    def build(cls, segments, k1, n_rows, device):
        zero = n_rows
        past_segments = [i for i, s in enumerate(segments) if s.past]
        virtual = []
        for i, s in enumerate(segments):
            if s.past:
                base = n_rows + 1 + k1 * past_segments.index(i)
                hist = list(range(base, base + k1))
            else:
                hist = [zero] * k1
            virtual.append(hist + list(range(s.start, s.start + s.trunk)))

        def phase(runs):  # runs: (history rows, token rows)
            if not runs:
                return None
            ext, take, rows, lens, at = [], [], [], [], 0
            for hist, toks in runs:
                ext += hist + toks
                take += range(at, at + len(toks))
                rows += toks
                lens.append(len(toks))
                at += len(hist) + len(toks)
            return Phase(to_device(ext, device), to_device(take, device), to_device(rows, device), lens,
                         cu_seqlens(lens, device))

        t1, t2, split, saves, br, br_seg = [], [], [], [], [], []
        for i, s in enumerate(segments):
            v = virtual[i]
            cut = s.save if 0 < s.save < s.trunk else s.trunk
            t1.append((v[:k1], v[k1:k1 + cut]))
            if cut < s.trunk:
                t2.append((v[cut:k1 + cut], v[k1 + cut:]))
                split.append(i)
            if s.save:
                saves.append((i, torch.tensor(v[cut:k1 + cut], dtype=torch.long, device=device)))
            for a, b in s.branches:
                br.append((v[-k1:], list(range(a, b))))
                br_seg.append(i)
        return cls(n_rows, phase(t1), phase(t2), phase(br), to_device(split, device) if split else None,
                   to_device(br_seg, device) if br_seg else None, past_segments, saves)


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

    def merge(self):
        self.qkv_split = merge_linears(self, ["q_proj", "k_proj", "v_proj"], "qkv_proj")

    def forward(self, x, cos, sin, segments):
        n = x.shape[0]
        if hasattr(self, "qkv_proj"):
            qg, kx, vx = self.qkv_proj(x).split(self.qkv_split, dim=-1)
        else:
            qg, kx, vx = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q, gate = qg.reshape(n, self.h, 2 * self.d).chunk(2, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(kx.reshape(n, self.hkv, self.d))
        v = vx.reshape(n, self.hkv, self.d)
        q, k = rotate_partial(q, cos, sin), rotate_partial(k, cos, sin)
        o = attend(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), segments, self.index)
        o = sigmoid_gate(o.transpose(0, 1).reshape(n, self.h * self.d), gate.reshape(n, -1))
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def merge(self):
        merge_linears(self, ["gate_proj", "up_proj"], "gate_up_proj")

    def forward(self, x):
        if hasattr(self, "gate_up_proj"):
            return self.down_proj(swiglu(*self.gate_up_proj(x).chunk(2, dim=-1)))
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


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

    def merge_projections(self):
        """Serve-time: one matmul per group of projections that share an input."""
        for layer in self.model.layers:
            (layer.linear_attn if layer.kind == "linear_attention" else layer.self_attn).merge()
            layer.mlp.merge()
        return self

    def hidden(self, ids, positions, segments):
        ang = positions.float()[:, None] * self.inv_freq[None, :]
        ang = torch.cat((ang, ang), dim=-1)
        x = self.model.embed_tokens(ids)
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        for layer in self.model.layers:
            x = layer(x, cos, sin, segments)
        return self.model.norm(x)

    def label_logits(self, h, label_ids):
        return self.lm_head.weight[label_ids].float() @ h.float()  # fp32: bf16 logits step by 0.06+
