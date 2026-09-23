"""Packed, prefill-only reads on dan's own model code.

Any number of read plans run in one forward pass: each plan contributes its
prompt (trunk) followed by one branch per question; attention runs per plan
under a tree mask (see ``attention``). Branch positions continue from the end
of their trunk, so every question sees the positions it would in a separate
forward. A plan whose static prefix is cached starts from its keys and values
and computes only the rest of its prompt.
"""
import torch

from .attention import segment
from .cache import PrefixCache
from .models import load


class Runner:
    def __init__(self, model, device="cpu", dtype=torch.float32, cache_bytes=2 << 30, adapter=None,
                 quantization=None, merge_projections=True):
        """``merge_projections``: serve-time fusion of projections that share an
        input; training (LoRA on the separate projections) passes False."""
        self.model = load(model, device, dtype)
        if adapter:
            from .lora import load_merged

            load_merged(self.model, adapter)
        if merge_projections:
            self.model.merge_projections()
        if quantization == "fp8":
            from .quant import quantize_fp8

            quantize_fp8(self.model)
        elif quantization:
            raise ValueError(f"unknown quantization {quantization!r}; use 'fp8'")
        self.device = device
        self.cache = PrefixCache(cache_bytes) if cache_bytes else None

    def pack(self, plans, use_cache=True):
        """Flat ids and positions, one segment per plan with questions, the
        read positions with their label ids, and the static prefixes to store."""
        ids, pos, segs, reads, stores = [], [], [], [], []
        use_cache = use_cache and self.cache is not None
        for plan in plans:
            if not plan.branches:
                continue
            static = plan.prefix[:plan.static]
            past = self.cache.get(static) if use_cache and static else None
            skip = plan.static if past else 0
            start = len(ids)
            trunk = plan.prefix[skip:]
            ids += trunk
            pos += range(skip, len(plan.prefix))
            branch = [0] * len(trunk)
            for j, b in enumerate(plan.branches, 1):
                ids += b.suffix
                pos += range(len(plan.prefix), len(plan.prefix) + len(b.suffix))
                branch += [j] * len(b.suffix)
                reads.append((len(ids) - 1, b.label_ids))
            save = plan.static if use_cache and static and not past else 0
            segs.append(segment(start, branch, past, save, skip, self.device))
            if save:
                stores.append((static, segs[-1]))
        t = lambda xs: torch.tensor(xs, dtype=torch.long, device=self.device)
        return t(ids), t(pos), segs, reads, stores

    def forward(self, plans, use_cache=True):
        """Per plan, per branch: the logits of its label tokens, on the model's
        device and differentiable (training passes ``use_cache=False``)."""
        ids, pos, segs, reads, stores = self.pack(plans, use_cache)
        flat = []
        if reads:
            h = self.model.hidden(ids, pos, segs)
            rows = h[torch.tensor([i for i, _ in reads], device=self.device)]
            flat = [self.model.label_logits(r, labels) for r, (_, labels) in zip(rows, reads)]
            for static, seg in stores:
                self.cache.put(static, seg.saved)
        return self._split(plans, flat)

    @torch.inference_mode()
    def read_many(self, plans):
        """Per plan, per branch: the logits of its label tokens (fp32, CPU)."""
        return [[x.float().cpu() for x in per] for per in self.forward(plans)]

    @staticmethod
    def _split(plans, flat):
        out, k = [], 0
        for plan in plans:
            n = len(plan.branches)
            out.append(flat[k:k + n])
            k += n
        return out

    def read(self, plan):
        return self.read_many([plan])[0]
