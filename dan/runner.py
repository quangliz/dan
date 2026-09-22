"""Packed, prefill-only reads on dan's own model code.

Any number of read plans run in one forward pass: each plan contributes its
prompt (trunk) followed by one branch per question, and a tree mask keeps
requests apart and questions independent (see ``attention.tree_mask``). Branch
positions continue from the end of their trunk, so every question sees the
same positions it would in a separate forward.
"""
import torch

from .attention import tree_mask
from .models import load


class Runner:
    def __init__(self, model, device="cpu", dtype=torch.float32):
        self.model = load(model, device, dtype)
        self.device = device

    def pack(self, plans):
        ids, pos, req, branch, reads = [], [], [], [], []
        for r, plan in enumerate(plans):
            if not plan.branches:
                continue
            p = len(plan.prefix)
            ids += plan.prefix
            pos += range(p)
            req += [r] * p
            branch += [0] * p
            for j, b in enumerate(plan.branches, 1):
                ids += b.suffix
                pos += range(p, p + len(b.suffix))
                req += [r] * len(b.suffix)
                branch += [j] * len(b.suffix)
                reads.append((len(ids) - 1, b.label_ids))
        t = lambda xs: torch.tensor(xs, dtype=torch.long, device=self.device)
        return t(ids), t(pos), t(req), t(branch), reads

    @torch.inference_mode()
    def read_many(self, plans):
        """Per plan, per branch: the logits of its label tokens."""
        ids, pos, req, branch, reads = self.pack(plans)
        flat = []
        if reads:
            h = self.model.hidden(ids, pos, tree_mask(req, branch))
            flat = [self.model.label_logits(h[i], labels).float().cpu() for i, labels in reads]
        out, k = [], 0
        for plan in plans:
            n = len(plan.branches)
            out.append(flat[k:k + n])
            k += n
        return out

    def read(self, plan):
        return self.read_many([plan])[0]
