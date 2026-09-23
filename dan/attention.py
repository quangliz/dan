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
from itertools import accumulate

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right


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


def segment(start, branch, past=None, save=0, past_len=0, device="cpu"):
    """A segment over new tokens with ``branch`` ids (a list: trunk first, then
    each branch contiguous), after ``past_len`` cached tokens."""
    mask = tree_mask(torch.tensor(branch, device=device))
    if past:
        mask = torch.cat([mask.new_ones(mask.shape[0], past_len), mask], dim=1)
    trunk = next((i for i, b in enumerate(branch) if b), len(branch))
    spans, i = [], trunk
    while i < len(branch):
        j = i
        while j < len(branch) and branch[j] == branch[i]:
            j += 1
        spans.append((start + i, start + j))
        i = j
    return Segment(start, start + len(branch), mask, trunk, spans, past, save)


def attend(q, k, v, segments, layer, scale=None):
    """q: [H, N, D]; k, v: [Hkv, N, D] (rotary applied). Returns [H, N, D].

    Per segment, the trunk is plain causal attention (flash kernels, native
    GQA; aligned bottom-right after cached tokens) and only the branch rows use
    the explicit tree mask. Fused kernels need 4-D inputs; the masked one also
    needs keys repeated per query head."""
    rep = q.shape[0] // k.shape[0]
    out = torch.empty_like(q)
    for s in segments:
        ks, vs = k[:, s.start:s.end], v[:, s.start:s.end]
        if s.save:
            s.saved[layer] = (ks[:, :s.save].clone(), vs[:, :s.save].clone())
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
            kb, vb = repeat_heads(ks.transpose(0, 1), rep).transpose(0, 1), repeat_heads(vs.transpose(0, 1), rep).transpose(0, 1)
            out[:, mid:s.end] = F.scaled_dot_product_attention(
                q[None, :, mid:s.end], kb[None], vb[None], attn_mask=s.mask[None, None, t:], scale=scale)[0]
    return out


def repeat_heads(x, rep):
    """[N, H, D] -> [N, H * rep, D], each head repeated ``rep`` times in place
    (like ``repeat_interleave`` on dim 1, without its device sync)."""
    if rep == 1:
        return x
    n, h, d = x.shape
    return x[:, :, None].expand(n, h, rep, d).reshape(n, h * rep, d)


def cu_seqlens(lengths, device):
    """Cumulative sequence lengths on the device and on the host (so varlen
    kernels need not copy them back, which would stall the CPU)."""
    host = torch.tensor([0, *accumulate(lengths)], dtype=torch.long)
    if torch.device(device).type == "cuda":
        return host.pin_memory().to(device, non_blocking=True), host  # pinned: the upload does not block
    return host.to(device), host
