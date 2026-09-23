"""The packed tree-mask runner must match the per-question HF oracle."""
import pytest
import torch
from test_prompt_llm import QUESTIONS
from transformers import AutoTokenizer

from dan.attention import branch_rows_mask, tree_mask
from dan.prompt import Planner
from dan.reference import HFReference
from dan.runner import Runner

STATES = ["Where is the train station?", {"msg": "Ich liebe dieses Produkt!", "channel": "email"}, "Refund me now."]
EXTRA = {"urgent": {"type": "noul", "instructions": "Is it urgent?", "criteria": {"true": "needs action today"}}}


def test_tree_mask():
    m = tree_mask(torch.tensor([0, 0, 1, 1, 2]))
    assert m[1].tolist() == [True, True, False, False, False]
    assert m[3].tolist() == [True, True, True, True, False]
    assert m[4].tolist() == [True, True, False, False, True]
    b = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3])
    assert torch.equal(branch_rows_mask(b, 3), tree_mask(b)[3:])


@pytest.mark.parametrize("model", ["tiny", "small"])
def test_parity_with_reference(model, request):
    name = request.getfixturevalue(model)
    planner = Planner(AutoTokenizer.from_pretrained(name))
    plans = [planner.plan(s, dict(QUESTIONS, **(EXTRA if i % 2 else {}))) for i, s in enumerate(STATES)]
    ref = HFReference(name)
    expected = [ref.read(p) for p in plans]
    runner = Runner(name)
    # cold cache, then warm (static prefixes reused), then a mixed batch
    for batch in (plans, plans, [plans[1], planner.plan("new state", QUESTIONS), plans[0]]):
        got = runner.read_many(batch)
        for plan, logits in zip(batch, got):
            want = expected[plans.index(plan)] if plan in plans else ref.read(plan)
            for a, b in zip(logits, want):
                torch.testing.assert_close(a, b, atol=2e-3, rtol=1e-4)
    assert runner.cache.hits >= len(plans) + 2
    assert all(0 < p.static < len(p.prefix) for p in plans)
