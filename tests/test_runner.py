"""The packed tree-mask runner must match the per-question HF oracle."""
import pytest
import torch
from test_prompt_llm import QUESTIONS
from transformers import AutoTokenizer

from dan.attention import tree_mask
from dan.prompt import Planner
from dan.reference import HFReference
from dan.runner import Runner

STATES = ["Where is the train station?", {"msg": "Ich liebe dieses Produkt!", "channel": "email"}, "Refund me now."]
EXTRA = {"urgent": {"type": "noul", "instructions": "Is it urgent?", "criteria": {"true": "needs action today"}}}


def test_tree_mask():
    req = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    br = torch.tensor([0, 0, 1, 2, 0, 1, 1])
    m = tree_mask(req, br)
    assert m[2].tolist() == [True, True, True, False, False, False, False]
    assert m[3].tolist() == [True, True, False, True, False, False, False]
    assert m[6].tolist() == [False, False, False, False, True, True, True]


@pytest.mark.parametrize("model", ["tiny", "small"])
def test_parity_with_reference(model, request):
    name = request.getfixturevalue(model)
    planner = Planner(AutoTokenizer.from_pretrained(name))
    plans = [planner.plan(s, dict(QUESTIONS, **(EXTRA if i % 2 else {}))) for i, s in enumerate(STATES)]
    ref = HFReference(name)
    got = Runner(name).read_many(plans)
    for plan, logits in zip(plans, got):
        for a, b in zip(logits, ref.read(plan)):
            torch.testing.assert_close(a, b, atol=2e-3, rtol=1e-4)
