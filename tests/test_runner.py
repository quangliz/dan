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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen attention runs on CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_varlen_attention_matches_dense(dtype):
    """The block-sparse varlen path equals the dense tree-mask reference."""
    from dan import attention as A

    if A.varlen_attn is None:
        pytest.skip("torch without varlen attention")
    torch.manual_seed(0)
    h, hkv, d, layer = 8, 2, 128, 0
    # (trunk, [branch lengths], cached prefix)
    shapes = [(40, [5, 9, 1], 0), (7, [30], 12), (1, [3, 3, 3, 3, 3, 3], 0), (64, [], 0), (13, [17, 2], 20)]
    segs, start = A.Batch(), 0
    for trunk, lens, past in shapes:
        branch = [0] * trunk + [j for j, n in enumerate(lens, 1) for _ in range(n)]
        pkv = {layer: (torch.randn(hkv, past, d, device="cuda", dtype=dtype),
                       torch.randn(hkv, past, d, device="cuda", dtype=dtype))} if past else None
        segs.append(A.segment(start, branch, pkv, 0, past, "cuda"))
        start += len(branch)
    q = torch.randn(h, start, d, device="cuda", dtype=dtype)
    k = torch.randn(hkv, start, d, device="cuda", dtype=dtype)
    v = torch.randn(hkv, start, d, device="cuda", dtype=dtype)
    fast = A._attend_varlen(q, k, v, segs, layer, None)
    ref = A._attend_dense(q, k, v, segs, layer, None)
    torch.testing.assert_close(fast.float(), ref.float(), atol=2e-2, rtol=2e-2)
