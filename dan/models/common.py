"""Helpers shared by model implementations."""
import torch
from torch import nn


def merge_linears(parent, names, merged):
    """Replace ``parent``'s linears ``names`` (same input) with one linear
    ``merged`` whose output is theirs concatenated; returns the split sizes.
    One larger matmul instead of several keeps the GPU busier."""
    parts = [getattr(parent, n) for n in names]
    weight = torch.cat([p.weight.detach() for p in parts])
    bias = None
    if any(p.bias is not None for p in parts):
        bias = torch.cat([p.bias.detach() if p.bias is not None else weight.new_zeros(p.out_features) for p in parts])
    lin = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device=weight.device, dtype=weight.dtype)
    lin.weight = nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        lin.bias = nn.Parameter(bias, requires_grad=False)
    for n in names:
        delattr(parent, n)
    setattr(parent, merged, lin)
    return [p.out_features for p in parts]
