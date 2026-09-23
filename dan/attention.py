"""Tree attention for packed, prefill-only reads.

Many requests are packed into one flat token sequence so every projection and
MLP runs as one batched matmul; attention runs per request (a "segment").

Within a segment, tokens are a trunk (the request's prompt) followed by one
branch per question. A query may see a key when

    key comes first  and  (key is trunk or same branch)

so every question reads the whole prompt but none of the other questions. A
segment may start from cached keys and values of its prompt's leading tokens
(``past``); those are trunk, visible to every query.

Two exact implementations:

- **varlen** (CUDA, bf16/fp16): a few flash-attention calls over the whole
  packed batch, never touching masked-out pairs. Trunks: causal, on
  ``[past; trunk]`` keys. Branches: (A) every branch token against its
  segment's ``[past; trunk]`` keys, unmasked, and (B) every branch causally on
  its own keys; A and B are merged by their log-sum-exp. Cost follows what
  each branch actually reads, instead of branch tokens x all tokens.
- **dense** (CPU, fp32, or without varlen kernels): per segment, flash causal
  attention for the trunk and one masked call for the branch rows. It is the
  reference the varlen path is tested against.

Recurrent layers (see ``models.qwen3_5``) do not use attention: they run the
trunk, then every branch from the trunk's final state, using the segment's
``trunk`` length and ``branches`` spans. ``past`` and ``saved`` map a layer
index to that layer's cache entry, whatever its kind.
"""
import inspect
import os
from dataclasses import dataclass, field
from itertools import accumulate

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from .fused import lse_merge

try:  # flash varlen attention (torch >= 2.9)
    from torch.nn.attention.varlen import AuxRequest, varlen_attn

    _VARLEN_PARAMS = set(inspect.signature(varlen_attn).parameters)
except ImportError:  # pragma: no cover - older torch
    varlen_attn = AuxRequest = None
    _VARLEN_PARAMS = set()

VARLEN = os.environ.get("DAN_VARLEN", "1") != "0"


def tree_mask(branch):
    """[N, N] boolean mask (True = attend) from per-token branch ids (0 = trunk)."""
    n = branch.shape[0]
    idx = torch.arange(n, device=branch.device)
    causal = idx[None, :] <= idx[:, None]
    return causal & ((branch[None, :] == 0) | (branch[None, :] == branch[:, None]))


def branch_rows_mask(branch, trunk):
    """The tree mask's rows for branch tokens only, [N - trunk, N]: each sees
    the whole trunk and its own branch up to itself (== tree_mask(branch)[trunk:])."""
    n = branch.shape[0]
    cols = torch.arange(n, device=branch.device)
    rows = cols[trunk:]
    own = (branch[None, :] == branch[trunk:, None]) & (cols[None, :] <= rows[:, None])
    return (cols[None, :] < trunk) | own


@dataclass
class Segment:
    start: int  # this request's new tokens are packed[start:end]
    end: int
    trunk: int = 0  # new trunk tokens: packed[start:start + trunk]
    branches: list = field(default_factory=list)  # (start, end) of each branch in the packed sequence
    past: dict | None = None  # layer -> cache entry, e.g. (k, v) each [Hkv, past, D]
    save: int = 0  # keep the cache entry after the first ``save`` new tokens
    saved: dict = field(default_factory=dict)
    past_len: int = 0
    branch_ids: list = field(default_factory=list, repr=False)
    device: str = "cpu"
    _mask: torch.Tensor | None = field(default=None, repr=False)

    @property
    def mask(self):
        """Branch rows of the tree mask, [n - trunk, past + n]; built on first
        use (only the dense path needs it)."""
        if self._mask is None:
            m = branch_rows_mask(torch.tensor(self.branch_ids, device=self.device), self.trunk)
            if self.past:
                m = torch.cat([m.new_ones(m.shape[0], self.past_len), m], dim=1)
            self._mask = m
        return self._mask


class Batch(list):
    """The segments of one packed forward; caches their varlen layout, which
    is the same for every attention layer."""

    layout = None


def segment(start, branch, past=None, save=0, past_len=0, device="cpu"):
    """A segment over new tokens with ``branch`` ids (a list: trunk first, then
    each branch contiguous), after ``past_len`` cached tokens."""
    trunk = next((i for i, b in enumerate(branch) if b), len(branch))
    spans, i = [], trunk
    while i < len(branch):
        j = i
        while j < len(branch) and branch[j] == branch[i]:
            j += 1
        spans.append((start + i, start + j))
        i = j
    return Segment(start, start + len(branch), trunk, spans, past, save, past_len=past_len if past else 0,
                   branch_ids=branch, device=device)


def cu_seqlens(lengths, device, dtype=torch.long):
    """Cumulative sequence lengths on the device and on the host (so varlen
    kernels need not copy them back, which would stall the CPU)."""
    host = torch.tensor([0, *accumulate(lengths)], dtype=dtype)
    if torch.device(device).type == "cuda":
        return host.pin_memory().to(device, non_blocking=True), host  # pinned: the upload does not block
    return host.to(device), host


@dataclass
class ReadLayout:
    """Packed-row indices and cumulative lengths for the varlen path."""

    trunk_rows: torch.Tensor  # packed rows of every trunk, segment by segment
    branch_rows: torch.Tensor  # packed rows of every branch token, segment by segment
    cu_trunk_q: torch.Tensor  # per segment: its new trunk tokens
    cu_trunk_k: torch.Tensor  # per segment: past + new trunk tokens
    max_trunk_q: int
    max_trunk_k: int
    cu_branch_q: torch.Tensor  # per segment: all its branch tokens (part A queries)
    max_branch_q: int
    cu_branch: torch.Tensor  # per branch (part B)
    max_branch: int
    has_past: bool
    has_branches: bool

    @classmethod
    def build(cls, segments, device):
        trunk_rows, branch_rows = [], []
        tq, tk, bq, bl = [], [], [], []
        for s in segments:
            trunk_rows += range(s.start, s.start + s.trunk)
            branch_rows += range(s.start + s.trunk, s.end)
            tq.append(s.trunk)
            tk.append(s.past_len + s.trunk)
            bq.append(s.end - s.start - s.trunk)
            bl += [b - a for a, b in s.branches]

        def idx(rows):
            return torch.tensor(rows, dtype=torch.long).pin_memory().to(device, non_blocking=True)

        i32 = torch.int32
        return cls(idx(trunk_rows), idx(branch_rows),
                   cu_seqlens(tq, device, i32)[0], cu_seqlens(tk, device, i32)[0], max(tq), max(tk),
                   cu_seqlens(bq, device, i32)[0], max(bq),
                   cu_seqlens(bl, device, i32)[0] if bl else None, max(bl, default=0),
                   any(s.past for s in segments), bool(bl))


def use_varlen(q):
    return VARLEN and varlen_attn is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)


def _varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal, scale, lse=False):
    """q: [T, H, D]; k, v: [T, Hkv, D]."""
    rep = q.shape[1] // k.shape[1]
    kw = {"scale": scale}
    if "window_size" in _VARLEN_PARAMS:
        kw["window_size"] = (-1, 0) if causal else (-1, -1)
    else:  # older signature
        kw["is_causal"] = causal
    if rep > 1:
        if "enable_gqa" in _VARLEN_PARAMS:
            kw["enable_gqa"] = True
        else:
            k, v = repeat_heads(k, rep), repeat_heads(v, rep)
    if lse:
        kw["return_aux"] = AuxRequest(lse=True)
    out = varlen_attn(q, k, v, cu_q, cu_k, max_q, max_k, **kw)
    if lse:
        return out[0], out[1]
    return out


def attend(q, k, v, segments, layer, scale=None):
    """q: [H, N, D]; k, v: [Hkv, N, D] (rotary applied). Returns [H, N, D]."""
    for s in segments:
        if s.save:
            s.saved[layer] = (k[:, s.start:s.start + s.save].clone(), v[:, s.start:s.start + s.save].clone())
    if use_varlen(q):
        return _attend_varlen(q, k, v, segments, layer, scale)
    return _attend_dense(q, k, v, segments, layer, scale)


def _attend_varlen(q, k, v, segments, layer, scale):
    lay = getattr(segments, "layout", None)
    if lay is None:
        lay = ReadLayout.build(segments, q.device)
        if isinstance(segments, Batch):
            segments.layout = lay
    qn, kn, vn = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # [N, H, D]
    if lay.has_past:  # [past; trunk] keys per segment
        parts_k, parts_v = [], []
        for s in segments:
            if s.past:
                pk, pv = s.past[layer]
                parts_k.append(pk.transpose(0, 1))
                parts_v.append(pv.transpose(0, 1))
            parts_k.append(kn[s.start:s.start + s.trunk])
            parts_v.append(vn[s.start:s.start + s.trunk])
        kt, vt = torch.cat(parts_k), torch.cat(parts_v)
    else:
        kt, vt = kn.index_select(0, lay.trunk_rows), vn.index_select(0, lay.trunk_rows)
    out = qn.new_empty(qn.shape)
    qt = qn.index_select(0, lay.trunk_rows)
    out.index_copy_(0, lay.trunk_rows,
                    _varlen(qt, kt, vt, lay.cu_trunk_q, lay.cu_trunk_k, lay.max_trunk_q, lay.max_trunk_k, True, scale))
    if lay.has_branches:
        qb = qn.index_select(0, lay.branch_rows)
        kb, vb = kn.index_select(0, lay.branch_rows), vn.index_select(0, lay.branch_rows)
        o_a, lse_a = _varlen(qb, kt, vt, lay.cu_branch_q, lay.cu_trunk_k, lay.max_branch_q, lay.max_trunk_k,
                             False, scale, lse=True)
        o_b, lse_b = _varlen(qb, kb, vb, lay.cu_branch, lay.cu_branch, lay.max_branch, lay.max_branch,
                             True, scale, lse=True)
        out.index_copy_(0, lay.branch_rows, lse_merge(o_a, lse_a, o_b, lse_b))
    return out.transpose(0, 1)


def _attend_dense(q, k, v, segments, layer, scale):
    """Per segment: flash causal attention for the trunk (native GQA; aligned
    bottom-right after cached tokens) and one masked call for the branch rows.
    Fused kernels need 4-D inputs; the masked one also needs keys repeated per
    query head."""
    rep = q.shape[0] // k.shape[0]
    out = torch.empty_like(q)
    for s in segments:
        ks, vs = k[:, s.start:s.end], v[:, s.start:s.end]
        past = 0
        if s.past:
            pk, pv = s.past[layer]
            past = pk.shape[1]
            ks, vs = torch.cat([pk, ks], dim=1), torch.cat([pv, vs], dim=1)
        t, mid = s.trunk, s.start + s.trunk
        if t:
            kt, vt = ks[None, :, :past + t], vs[None, :, :past + t]
            mask = causal_lower_right(t, past + t) if past else None
            out[:, s.start:mid] = F.scaled_dot_product_attention(
                q[None, :, s.start:mid], kt, vt, attn_mask=mask, is_causal=not past, scale=scale,
                enable_gqa=rep > 1)[0]
        if s.end > mid:
            kb = repeat_heads(ks.transpose(0, 1), rep).transpose(0, 1)
            vb = repeat_heads(vs.transpose(0, 1), rep).transpose(0, 1)
            out[:, mid:s.end] = F.scaled_dot_product_attention(
                q[None, :, mid:s.end], kb[None], vb[None], attn_mask=s.mask[None, None], scale=scale)[0]
    return out


def repeat_heads(x, rep):
    """[N, H, D] -> [N, H * rep, D], each head repeated ``rep`` times in place
    (like ``repeat_interleave`` on dim 1, without its device sync)."""
    if rep == 1:
        return x
    n, h, d = x.shape
    return x[:, :, None].expand(n, h, rep, d).reshape(n, h * rep, d)
