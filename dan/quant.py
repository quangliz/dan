"""FP8 (e4m3) weights and activations for decoder linears, on GPUs with FP8
tensor cores (Ada sm89, Hopper sm90+).

Weights are quantized once and activations on the fly. On Hopper and newer,
scales are per output row and per token; on Ada (sm89), row-wise scaled
matmuls take a slow path (26 vs 93 TFLOPS on an L4), so both scales are one
per tensor there. Embeddings and the label head stay in the
model dtype: they are gathers, not matmuls, on the read path.
"""
import torch
from torch import nn

E4M3 = torch.float8_e4m3fn
E4M3_MAX = torch.finfo(E4M3).max


def supported(device=None):
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device) >= (8, 9)


class FP8Linear(nn.Module):
    def __init__(self, linear, rowwise=True):
        super().__init__()
        w = linear.weight.detach()
        self.out_features, self.in_features = w.shape
        amax = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) if rowwise else \
            w.abs().amax().float().clamp(min=1e-12).reshape(1, 1)
        scale = amax / E4M3_MAX
        self.weight = nn.Parameter((w.float() / scale).to(E4M3), requires_grad=False)  # [out, in]
        self.register_buffer("weight_scale", scale.reshape(1, -1) if rowwise else scale)  # [1, out] or [1, 1]
        self.bias = linear.bias
        self.rowwise = rowwise
        self.out_dtype = w.dtype

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        if self.rowwise:
            amax = x.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12)  # [tokens, 1]
        else:
            amax = x.abs().amax().float().clamp(min=1e-12).reshape(1, 1)
        scale = amax / E4M3_MAX
        xq = (x.float() / scale).to(E4M3)
        y = torch._scaled_mm(xq, self.weight.t(), scale_a=scale, scale_b=self.weight_scale,
                             bias=self.bias, out_dtype=self.out_dtype)
        return y.reshape(*shape[:-1], self.out_features)


def rowwise_ok(device):
    """Row-wise scaled FP8 matmul: fast on sm90+, slow on sm89, and support
    varies by torch version."""
    if torch.cuda.get_device_capability(device) < (9, 0):
        return False
    try:
        a = torch.ones(16, 32, device=device).to(E4M3)
        b = torch.ones(16, 32, device=device).to(E4M3)
        torch._scaled_mm(a, b.t(), scale_a=torch.ones(16, 1, device=device), scale_b=torch.ones(1, 16, device=device),
                         out_dtype=torch.bfloat16)
        return True
    except (RuntimeError, NotImplementedError):
        return False


def quantize_fp8(model, min_features=16):
    """Replace the decoder's nn.Linear modules with FP8Linear in place."""
    device = next(model.parameters()).device
    if not supported(device):
        raise RuntimeError("FP8 needs a CUDA GPU with compute capability 8.9 or newer")
    rowwise = rowwise_ok(device)
    for layer in model.model.layers:
        for mod in list(layer.modules()):
            for name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear) and min(child.in_features, child.out_features) >= min_features \
                        and child.in_features % 16 == 0 and child.out_features % 16 == 0:
                    setattr(mod, name, FP8Linear(child, rowwise))
    torch.cuda.empty_cache()
    return model
