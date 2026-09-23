"""Llama-family decoder (Llama 2/3, Mistral, Qwen2/2.5/3, SmolLM), prefill only.

Module and parameter names mirror Hugging Face checkpoints so their
safetensors load directly. There is no KV cache and no full-vocabulary head:
the runner asks for hidden states at read positions and projects them onto
the label tokens alone.
"""
import math

import torch
from torch import nn

from ..attention import attend
from ..fused import rms_norm, swiglu
from .common import merge_linears


def rope_settings(config):
    params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    theta = params.get("rope_theta") or getattr(config, "rope_theta", None) or 10000.0
    return float(theta), params


def inv_frequencies(config, head_dim):
    theta, params = rope_settings(config)
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    kind = params.get("rope_type") or params.get("type") or "default"
    if kind == "default":
        return inv.float()
    if kind == "linear":
        return (inv / params["factor"]).float()
    if kind == "llama3":
        factor, lo, hi = params["factor"], params["low_freq_factor"], params["high_freq_factor"]
        old = params["original_max_position_embeddings"]
        wavelen = 2 * math.pi / inv
        smooth = (old / wavelen - lo) / (hi - lo)
        scaled = torch.where(wavelen > old / lo, inv / factor, inv)
        mid = (wavelen <= old / lo) & (wavelen >= old / hi)
        return torch.where(mid, (1 - smooth) * inv / factor + smooth * inv, scaled).float()
    raise NotImplementedError(f"rope type {kind!r} is not supported yet")


def rotate(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-b, a), dim=-1) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, c, head_dim, qk_norm):
        super().__init__()
        self.h, self.hkv, self.d = c.num_attention_heads, c.num_key_value_heads, head_dim
        bias = getattr(c, "attention_bias", False) or c.model_type == "qwen2"
        self.q_proj = nn.Linear(c.hidden_size, self.h * head_dim, bias=bias)
        self.k_proj = nn.Linear(c.hidden_size, self.hkv * head_dim, bias=bias)
        self.v_proj = nn.Linear(c.hidden_size, self.hkv * head_dim, bias=bias)
        self.o_proj = nn.Linear(self.h * head_dim, c.hidden_size, bias=getattr(c, "attention_bias", False))
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, c.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, c.rms_norm_eps)
        self.qk_norm = qk_norm

    def merge(self):
        self.qkv_split = merge_linears(self, ["q_proj", "k_proj", "v_proj"], "qkv_proj")

    def forward(self, x, cos, sin, segments, layer):
        n = x.shape[0]
        if hasattr(self, "qkv_proj"):
            q, k, v = self.qkv_proj(x).split(self.qkv_split, dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q, k, v = q.reshape(n, self.h, self.d), k.reshape(n, self.hkv, self.d), v.reshape(n, self.hkv, self.d)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = rotate(q, cos[:, None], sin[:, None]).transpose(0, 1)
        k = rotate(k, cos[:, None], sin[:, None]).transpose(0, 1)
        o = attend(q, k, v.transpose(0, 1), segments, layer)
        return self.o_proj(o.transpose(0, 1).reshape(n, self.h * self.d))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def merge(self):
        merge_linears(self, ["gate_proj", "up_proj"], "gate_up_proj")

    def forward(self, x):
        if hasattr(self, "gate_up_proj"):
            return self.down_proj(swiglu(*self.gate_up_proj(x).chunk(2, dim=-1)))
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Layer(nn.Module):
    def __init__(self, c, head_dim, qk_norm, index):
        super().__init__()
        self.index = index
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.self_attn = Attention(c, head_dim, qk_norm)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = MLP(c)

    def forward(self, x, cos, sin, segments):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, segments, self.index)
        return x + self.mlp(self.post_attention_layernorm(x))


class Body(nn.Module):
    def __init__(self, c, head_dim, qk_norm):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(Layer(c, head_dim, qk_norm, i) for i in range(c.num_hidden_layers))
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)


class LlamaForReads(nn.Module):
    MODEL_TYPES = frozenset({"llama", "mistral", "qwen2", "qwen3"})

    def __init__(self, config):
        super().__init__()
        self.config = config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.model = Body(config, head_dim, qk_norm=config.model_type == "qwen3")
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.register_buffer("inv_freq", inv_frequencies(config, head_dim), persistent=False)

    @staticmethod
    def frequencies(config):
        return inv_frequencies(config, getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)

    def merge_projections(self):
        """Serve-time: one matmul per group of projections that share an input."""
        for layer in self.model.layers:
            layer.self_attn.merge()
            layer.mlp.merge()
        return self

    def hidden(self, ids, positions, segments):
        """Final-normed hidden states for a packed token sequence whose
        requests are ``segments`` (see ``attention.Segment``)."""
        ang = positions.float()[:, None] * self.inv_freq[None, :]
        ang = torch.cat((ang, ang), dim=-1)
        x = self.model.embed_tokens(ids)
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        for layer in self.model.layers:
            x = layer(x, cos, sin, segments)
        return self.model.norm(x)

    def label_logits(self, h, label_ids):
        """Logits of just ``label_ids`` for one hidden state."""
        return self.lm_head.weight[label_ids].float() @ h.float()  # fp32: bf16 logits step by 0.06+
