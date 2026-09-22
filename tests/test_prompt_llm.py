import math

import pytest
from transformers import AutoTokenizer

from dan.prompt import Planner

QUESTIONS = {
    "is_question": {"type": "noul", "instructions": "Is the message a question?"},
    "language": {"type": "choice", "instructions": "Which language is the message written in?",
                 "criteria": {"english": None, "french": None, "german": None, "spanish": None}},
    "sentiment": {"type": "score", "instructions": "How positive is the message?",
                  "criteria": ["very negative", "neutral", "very positive"]},
}


@pytest.mark.parametrize("layout", ["inline", "system"])
@pytest.mark.parametrize("model", ["tiny", "small"])
def test_plan_slots_match_full_tokenization(model, layout, request):
    name = request.getfixturevalue(model)
    tok = AutoTokenizer.from_pretrained(name)
    planner = Planner(tok)
    state = "Where is the train station?"
    plan = planner.plan(state, QUESTIONS, layout=layout)
    assert len(plan.branches) == 3
    assert state in tok.decode(plan.prefix)
    for n, (b, labs) in enumerate(zip(plan.branches, plan.labels), 1):
        assert len(set(b.label_ids)) == len(labs)
        for lab, lid in zip(labs, b.label_ids):
            ids = plan.prefix + b.suffix + [lid]
            text = tok.decode(ids)
            if layout == "system":
                assert text.endswith(f"q{n}: {lab}")
            else:  # the branch holds its own question and the label ends the reply's first token
                assert text.endswith(lab) and QUESTIONS[list(QUESTIONS)[n - 1]]["instructions"] in tok.decode(b.suffix)
                spec = plan.specs[n - 1]
                full = planner.prefix_ids(None, f"{state}\n\n{planner.question_text(spec, labs)}")
                assert ids[:-1] == full  # trunk + branch is exactly the single-question prompt


def test_choice_labels_scale(tiny):
    tok = AutoTokenizer.from_pretrained(tiny)
    crit = {f"opt{i}": None for i in range(200)}
    plan = Planner(tok).plan("x", {"c": {"type": "choice", "criteria": crit}})
    assert len(set(plan.branches[0].label_ids)) == 200


def test_decide_shapes(small):
    """Answer shapes and invariants only: a 0.5B model is too weak to assert
    accuracy on (it reads French as English here). Accuracy belongs to evals."""
    from dan import LLM

    llm = LLM(small)
    out = llm.decide("Où se trouve la gare, s'il vous plaît ?", dict(QUESTIONS, one={"type": "choice", "criteria": {"x": None}}))
    assert list(out) == [*QUESTIONS, "one"]
    assert out["one"] == {"type": "choice", "choice": "x", "probabilities": {"x": 1.0}, "confidence": 1.0}
    assert 0.0 <= out["is_question"]["noul"] <= 1.0
    lang = out["language"]
    assert math.isclose(sum(lang["probabilities"].values()), 1.0, rel_tol=1e-6)
    assert lang["choice"] == max(lang["probabilities"], key=lang["probabilities"].get)
    assert 0.0 <= lang["confidence"] <= 1.0
    assert 0.0 <= out["sentiment"]["score"] <= 2.0
