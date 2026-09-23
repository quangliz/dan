"""Tree attention for packed, prefill-only reads.

Many requests are packed into one flat token sequence so every projection and
MLP runs as one batched matmul; attention alone runs per request (a
"segment"), so cost grows with each request's own length rather than the
square of the whole batch.

Within a segment, tokens are a trunk (the request's prompt) followed by one
branch per question. A query may see a key when

    key comes first  and  (key is trunk or same branch)

so every question reads the whole prompt but none of the other questions. A
segment may start from cached keys and values of its prompt's leading tokens
(``past``); those are trunk, visible to every query.

Recurrent layers (see ``models.qwen3_5``) do not use the mask: they run the
trunk, then every branch from the trunk's final state, using the segment's
``trunk`` length and ``branches`` spans. ``past`` and ``saved`` map a layer
index to that layer's cache entry, whatever its kind.
"""
from dataclasses import dataclass, field

import torch


def tree_mask(branch):
    """[N, N] boolean mask (True = attend) from per-token branch ids (0 = trunk)."""
    n = branch.shape[0]
    idx = torch.arange(n, device=branch.device)
    causal = idx[None, :] <= idx[:, None]
    return causal & ((branch[None, :] == 0) | (branch[None, :] == branch[:, None]))


@dataclass
class Segment:
    start: int  # this request's new tokens are packed[start:end]
    end: int
    mask: torch.Tensor  # [n, past + n]
    trunk: int = 0  # new trunk tokens: packed[start:start + trunk]
    branches: list = field(default_factory=list)  # (start, end) of each branch in the packed sequence
    past: dict | None = None  # layer -> cache entry, e.g. (k, v) each [Hkv, past, D]
    save: int = 0  # keep the cache entry after the first ``save`` new tokens
    saved: dict = field(default_factory=dict)


def segment(start, branch, past=None, save=0, past_len=0):
    """A segment over new tokens with ``branch`` ids (trunk first, then each
    branch contiguous), after ``past_len`` cached tokens."""
    mask = tree_mask(branch)
    if past:
        mask = torch.cat([mask.new_ones(mask.shape[0], past_len), mask], dim=1)
    ids = branch.tolist()
    trunk = next((i for i, b in enumerate(ids) if b), len(ids))
    spans, i = [], trunk
    while i < len(ids):
        j = i
        while j < len(ids) and ids[j] == ids[i]:
            j += 1
        spans.append((start + i, start + j))
        i = j
    return Segment(start, start + len(ids), mask, trunk, spans, past, save)


def attend(q, k, v, segments, layer, scale=None):
    """q: [H, N, D]; k, v: [Hkv, N, D] (rotary applied). Returns [H, N, D]."""
    rep = q.shape[0] // k.shape[0]
    out = torch.empty_like(q)
    for s in segments:
        ks, vs = k[:, s.start:s.end], v[:, s.start:s.end]
        if s.save:
            s.saved[layer] = (ks[:, :s.save].clone(), vs[:, :s.save].clone())
        if s.past:
            pk, pv = s.past[layer]
            ks, vs = torch.cat([pk, ks], dim=1), torch.cat([pv, vs], dim=1)
        if rep > 1:
            ks, vs = ks.repeat_interleave(rep, dim=0), vs.repeat_interleave(rep, dim=0)
        out[:, s.start:s.end] = torch.nn.functional.scaled_dot_product_attention(
            q[:, s.start:s.end], ks, vs, attn_mask=s.mask, scale=scale)
    return out
