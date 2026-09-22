import math

import pytest

from dan.prompt import ReadPlan
from dan.readout import answers, confidence
from dan.schema import SchemaError, parse_questions


def test_forced_and_specs():
    specs, forced = parse_questions({
        "a": {"type": "noul", "instructions": "Is it urgent?"},
        "b": {"type": "choice", "criteria": {"only": None}},
        "c": {"type": "score", "criteria": ["low"]},
        "d": {"type": "choice", "criteria": {"x": "desc", "y": {"k": 1}}},
    })
    assert [s.key for s in specs] == ["a", "d"]
    assert specs[1].options == [("x", "desc"), ("y", '{"k": 1}')]
    assert forced["b"]["choice"] == "only" and forced["c"]["score"] == 0.0


def test_limits():
    with pytest.raises(SchemaError):
        parse_questions({"s": {"type": "score", "criteria": [str(i) for i in range(11)]}})
    with pytest.raises(SchemaError):
        parse_questions({"c": {"type": "choice", "criteria": {}}})


def test_confidence():
    assert confidence([1.0, 0.0]) == 1.0
    assert math.isclose(confidence([0.25] * 4), 0.0, abs_tol=1e-12)


def test_answers_shapes_and_order():
    specs, forced = parse_questions({
        "n": {"type": "noul"},
        "f": {"type": "choice", "criteria": {"only": None}},
        "s": {"type": "score", "criteria": ["lo", "mid", "hi"]},
    })
    plan = ReadPlan([], specs, [], [], forced, ["n", "f", "s"])
    out = answers(plan, [[0.0, 0.0], [0.0, 0.0, 10.0]])
    assert list(out) == ["n", "f", "s"]
    assert math.isclose(out["n"]["noul"], 0.5)
    assert out["s"]["score"] > 1.99 and out["s"]["legend"] == {"0": "lo", "1": "mid", "2": "hi"}
