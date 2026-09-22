"""Tree masks for packed, prefill-only reads.

Many requests are packed into one flat token sequence. Each request is a trunk
(its shared prompt) followed by one branch per question. Every token carries
its request id and branch id (0 for the trunk), and a query may see a key when

    same request  and  key comes first  and  (key is trunk or same branch)

so every question reads the whole prompt but none of the other questions.
"""
import torch


def tree_mask(req, branch):
    """Dense [N, N] boolean mask (True = attend) from per-token ids."""
    n = req.shape[0]
    idx = torch.arange(n, device=req.device)
    causal = idx[None, :] <= idx[:, None]
    same_req = req[None, :] == req[:, None]
    visible = (branch[None, :] == 0) | (branch[None, :] == branch[:, None])
    return causal & same_req & visible


def attend(q, k, v, mask):
    """q: [H, N, D]; k, v: [Hkv, N, D]; grouped-query attention under ``mask``."""
    rep = q.shape[0] // k.shape[0]
    if rep > 1:
        k = k.repeat_interleave(rep, dim=0)
        v = v.repeat_interleave(rep, dim=0)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
