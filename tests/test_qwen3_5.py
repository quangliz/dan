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


def test_hybrid_varlen_matches_dense_bf16(hybrid, monkeypatch):
    """bf16 on CUDA: the varlen attention path gives the dense path's answers."""
    import pytest

    from dan import attention as A

    if not torch.cuda.is_available() or A.varlen_attn is None:
        pytest.skip("needs CUDA and varlen attention")
    planner = Planner(AutoTokenizer.from_pretrained(hybrid))
    plans = [planner.plan(s, dict(QUESTIONS, **EXTRA)) for s in STATES]
    runner = Runner(hybrid, "cuda", torch.bfloat16)
    fast = [runner.read_many(plans), runner.read_many(plans)]  # cold, then warm (cached prefixes)
    monkeypatch.setattr(A, "VARLEN", False)
    runner.cache.entries.clear()
    dense = runner.read_many(plans)
    for got in fast:
        for a_plan, b_plan in zip(got, dense):
            for a, b in zip(a_plan, b_plan):
                pa, pb = torch.softmax(a, 0), torch.softmax(b, 0)
                assert (pa - pb).abs().max() < 3e-2
