"""Qwen3.5 (hybrid recurrent + attention): tree reads and the prefix cache must
match the per-question Hugging Face oracle."""
import torch
from test_prompt_llm import QUESTIONS
from transformers import AutoTokenizer

from dan.prompt import Planner
from dan.reference import HFReference
from dan.runner import Runner

STATES = ["Where is the train station?", {"msg": "Ich liebe dieses Produkt!", "channel": "email"},
          "Refund me now. " * 20]  # > one 64-token delta-rule chunk
EXTRA = {"urgent": {"type": "noul", "instructions": "Is it urgent?"}}


def test_hybrid_parity_with_reference(hybrid, device):
    planner = Planner(AutoTokenizer.from_pretrained(hybrid))
    plans = [planner.plan(s, dict(QUESTIONS, **(EXTRA if i % 2 else {})), layout=layout)
             for i, s in enumerate(STATES) for layout in ("inline", "system")]
    ref = HFReference(hybrid, device)
    expected = [ref.read(p) for p in plans]
    runner = Runner(hybrid, device)
    # cold cache, warm cache (static prefixes reused), then a mixed batch
    for batch in (plans, plans, [plans[3], planner.plan("new state", QUESTIONS), plans[0]]):
        got = runner.read_many(batch)
        for plan, logits in zip(batch, got):
            want = expected[plans.index(plan)] if plan in plans else ref.read(plan)
            for a, b in zip(logits, want):
                torch.testing.assert_close(a, b, atol=5e-3, rtol=1e-3)
    assert runner.cache.hits >= len(plans)
