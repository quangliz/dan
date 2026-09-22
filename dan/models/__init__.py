"""Model registry and safetensors loading."""
import glob
import os

import torch

from .llama import LlamaForReads, inv_frequencies

ARCHS = [LlamaForReads]


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
    state = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        state.update(load_file(f))
    tied = getattr(config, "tie_word_embeddings", False)
    if tied:
        state.pop("lm_head.weight", None)
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    # RoPE frequencies were built on the meta device; rebuild them and keep them fp32
    inv_freq = inv_frequencies(config, model.inv_freq.shape[0] * 2)
    model.inv_freq = inv_freq
    model = model.to(device=device, dtype=dtype).eval()
    model.inv_freq = inv_freq.to(device)
    return model
