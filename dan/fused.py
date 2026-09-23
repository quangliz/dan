"""Elementwise pieces of the forward pass, fused by torch.compile on CUDA.

Each is a pure function over a few tensors (norms, gates, activations, FP8
activation scaling). Eager they are several passes over memory each; compiled
they become one kernel. Compilation is lazy, shape-dynamic (token counts vary
per batch) and CUDA-only; CPU runs them eagerly. Set DAN_COMPILE=0 to disable.
"""
import os

import torch
import torch.nn.functional as F

ENABLED = os.environ.get("DAN_COMPILE", "1") != "0"


def fused(fn):
    compiled = None

    def run(*args):
        nonlocal compiled
        if ENABLED and args[0].is_cuda:
            if compiled is None:
                compiled = torch.compile(fn, dynamic=True, fullgraph=True)
            return compiled(*args)
        return fn(*args)

    run.__name__, run.__doc__ = fn.__name__, fn.__doc__
    return run


@fused
def rms_norm(x, weight, eps):
    """Llama-style: normalize in fp32, cast back, then scale."""
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return weight * y.to(x.dtype)


@fused
def rms_norm_zero_centered(x, weight, eps):
    """Qwen3.5-style: the weight is stored as w and applied as (1 + w), in fp32."""
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + weight.float())).to(x.dtype)


@fused
def gated_rms_norm(x, gate, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    y = weight * y.to(x.dtype)
    return (y * F.silu(gate.float())).to(x.dtype)


@fused
def swiglu(gate, up):
    return F.silu(gate) * up


@fused
def sigmoid_gate(x, gate):
    return x * torch.sigmoid(gate)


@fused
def rotate_partial(x, cos, sin):
    """RoPE (rotate-half) on the first ``cos.shape[-1]`` dims of each head;
    x: [N, H, D], cos/sin: [N, R]."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    a, b = xr.chunk(2, dim=-1)
    return torch.cat([xr * cos[:, None] + torch.cat((-b, a), dim=-1) * sin[:, None], xp], dim=-1)


@fused
def delta_gates(b, a, a_log, dt_bias):
    """Gated DeltaNet write strength (beta) and log decay (g, fp32)."""
    return b.sigmoid(), -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())


@fused
def fp8_quantize(x, fp8_max, rowwise: bool):
    """Scale activations into e4m3 range: one scale per row, or one in all."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True) if rowwise else xf.abs().amax().reshape(1, 1)
    scale = amax.clamp(min=1e-12) / fp8_max
    return (xf / scale).to(torch.float8_e4m3fn), scale
