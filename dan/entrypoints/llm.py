"""Offline Python API.

    from dan import LLM
    llm = LLM("Qwen/Qwen2.5-0.5B-Instruct")
    llm.decide("I was charged twice", {"intent": {"type": "choice", "criteria": {"refund": None, "other": None}}})
"""
import torch

from ..prompt import Planner
from ..readout import answers


class LLM:
    def __init__(self, model, device="cpu", dtype=torch.float32, backend="dan"):
        """``backend``: "dan" (own packed tree-mask runner) or "reference"
        (a Hugging Face forward per question; the correctness oracle)."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.planner = Planner(self.tokenizer)
        if backend == "dan":
            from ..runner import Runner

            self.runner = Runner(model, device, dtype)
        elif backend == "reference":
            from ..reference import HFReference

            self.runner = HFReference(model, device, dtype)
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def decide(self, state, questions):
        """Answers keyed by question id, in Jev's answer shapes."""
        return self.decide_batch([(state, questions)])[0]

    def decide_batch(self, requests):
        """[(state, questions), ...] -> answers per request, in one forward
        pass on the dan backend."""
        plans = [self.planner.plan(s, q) for s, q in requests]
        if hasattr(self.runner, "read_many"):
            logits = self.runner.read_many(plans)
        else:
            logits = [self.runner.read(p) if p.branches else [] for p in plans]
        return [answers(p, lg) for p, lg in zip(plans, logits)]
