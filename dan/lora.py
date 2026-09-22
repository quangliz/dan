"""LoRA adapters for dan's own model code.

Training wraps the target linears with low-rank updates; serving merges them
into the base weights, so a tuned model reads exactly as fast as the base.
An adapter directory holds ``adapter.json`` (rank, alpha, targets) and
``adapter.safetensors`` (the A and B matrices, keyed by module path).
"""
import json
import math
import os

import torch
from torch import nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    def merged(self):
        self.base.weight.data += (self.lora_B @ self.lora_A).to(self.base.weight.dtype) * self.scale
        return self.base


def _targets(model, targets):
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in targets and isinstance(child, nn.Linear | LoRALinear):
                yield mod, child_name, f"{name}.{child_name}" if name else child_name


def inject(model, rank=16, alpha=32, targets=TARGETS):
    """Freeze the model and wrap its target linears; returns the trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for parent, child_name, _ in _targets(model, targets):
        wrapped = LoRALinear(getattr(parent, child_name), rank, alpha)
        setattr(parent, child_name, wrapped)
        params += [wrapped.lora_A, wrapped.lora_B]
    model.lora_config = {"rank": rank, "alpha": alpha, "targets": list(targets)}
    return params


def merge(model):
    for parent, child_name, _ in _targets(model, TARGETS):
        child = getattr(parent, child_name)
        if isinstance(child, LoRALinear):
            setattr(parent, child_name, child.merged())
    return model


def save(model, path):
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    tensors = {}
    for parent, child_name, full in _targets(model, TARGETS):
        child = getattr(parent, child_name)
        if isinstance(child, LoRALinear):
            tensors[f"{full}.lora_A"] = child.lora_A.detach().contiguous().cpu()
            tensors[f"{full}.lora_B"] = child.lora_B.detach().contiguous().cpu()
    save_file(tensors, os.path.join(path, "adapter.safetensors"))
    with open(os.path.join(path, "adapter.json"), "w") as f:
        json.dump(model.lora_config, f, indent=2)


def load_merged(model, path):
    """Apply the adapter at ``path`` to ``model`` and merge it in."""
    from safetensors.torch import load_file

    with open(os.path.join(path, "adapter.json")) as f:
        cfg = json.load(f)
    inject(model, cfg["rank"], cfg["alpha"], cfg["targets"])
    tensors = load_file(os.path.join(path, "adapter.safetensors"))
    for parent, child_name, full in _targets(model, cfg["targets"]):
        child = getattr(parent, child_name)
        child.lora_A.data.copy_(tensors[f"{full}.lora_A"])
        child.lora_B.data.copy_(tensors[f"{full}.lora_B"])
    return merge(model)
