"""Model registry and safetensors loading."""
import glob
import os

import torch

from .llama import LlamaForReads
from .qwen3_5 import Qwen35ForReads

ARCHS = [LlamaForReads, Qwen35ForReads]


def load(name, device="cpu", dtype=torch.float32):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(name)
    cls = next((a for a in ARCHS if config.model_type in a.MODEL_TYPES), None)
    if cls is None:
        raise NotImplementedError(f"model type {config.model_type!r} is not supported yet")
    path = name if os.path.isdir(name) else snapshot_download(name, allow_patterns=["*.json", "*.safetensors"])
    with torch.device("meta"):
        model = cls(config)
    remap = getattr(cls, "remap", lambda k: k)
    state = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        for k, v in load_file(f).items():
            k = remap(k)
            if k is not None:
                state[k] = v
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    if tied:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    # RoPE frequencies were built on the meta device; rebuild them and keep them fp32
    inv_freq = cls.frequencies(model.config)
    model.inv_freq = inv_freq
    model = model.to(device=device, dtype=dtype).eval()
    model.inv_freq = inv_freq.to(device)
    return model
