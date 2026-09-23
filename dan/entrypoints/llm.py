"""Offline Python API.

    from dan import LLM
    llm = LLM("Qwen/Qwen2.5-0.5B-Instruct")
    llm.decide("I was charged twice", {"intent": {"type": "choice", "criteria": {"refund": None, "other": None}}})
"""
import torch

from ..prompt import Planner
from ..readout import answers, combine


class LLM:
    def __init__(self, model, device="cpu", dtype=torch.float32, backend="dan", adapter=None, quantization=None):
        """``backend``: "dan" (own packed tree-mask runner) or "reference"
        (a Hugging Face forward per question; the correctness oracle).
        ``adapter``: a LoRA adapter directory from ``dan train`` (dan backend).
        ``quantization``: "fp8" for FP8 decoder weights and activations (sm89+)."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.planner = Planner(self.tokenizer)
        if backend == "dan":
            from ..runner import Runner

            self.runner = Runner(model, device, dtype, adapter=adapter, quantization=quantization)
        elif backend == "reference":
            if adapter or quantization:
                raise ValueError("adapters and quantization need the dan backend")
            from ..reference import HFReference

            self.runner = HFReference(model, device, dtype)
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def decide(self, state, questions, profile=None):
        """Answers keyed by question id, in Jev's answer shapes."""
        return self.decide_batch([(state, questions)], profile)[0]

    def decide_batch(self, requests, profile=None):
        """[(state, questions), ...] -> answers per request. On the dan backend
        all reads of all requests run in one forward pass."""
        groups, logp = self.read(requests, profile)
        return [answers(g[0], lp, profile, logits_canonical=True) for g, lp in zip(groups, logp)]

    def read(self, requests, profile=None, batch_size=None):
        """Per request: its read plans, and per question the log-probabilities
        in canonical option order, averaged over rotations, before the
        profile's correction."""
        groups = [self.planner.plans(s, q, profile) for s, q in requests]
        flat = [p for g in groups for p in g]
        step = batch_size or len(flat) or 1
        reads = []
        for i in range(0, len(flat), step):
            chunk = flat[i:i + step]
            if hasattr(self.runner, "read_many"):
                reads += self.runner.read_many(chunk)
            else:
                reads += [self.runner.read(p) if p.branches else [] for p in chunk]
        out, k = [], 0
        for g in groups:
            out.append(combine(g, reads[k:k + len(g)]))
            k += len(g)
        return groups, out
