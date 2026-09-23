import math

import pytest
import torch
from fastapi.testclient import TestClient
from test_prompt_llm import QUESTIONS
from transformers import AutoTokenizer

from dan import lora
from dan.calibration import Profile, apply, fit, metrics
from dan.engine import AsyncEngine
from dan.entrypoints.api_server import ServerSettings, create_app
from dan.prompt import Planner
from dan.readout import canonical, combine
from dan.runner import Runner
from dan.tuning import train_lora

TOPIC = {"topic": {"type": "choice", "instructions": "Topic?",
                   "criteria": {"sports": None, "business": None, "science": None, "world": None}}}


def test_metrics_and_temperature_fit():
    torch.manual_seed(0)
    n, k = 2000, 3
    true = torch.randn(n, k)
    y = torch.distributions.Categorical(logits=true).sample()
    over = torch.log_softmax(true * 3, 1)  # overconfident by 3x
    c = fit(over, y, "temperature")
    assert 2.5 < c["temperature"] < 3.5
    raw, cal = metrics(over, y), metrics(apply(over, c), y)
    assert cal["ece"] < raw["ece"] and cal["nll"] < raw["nll"]
    assert math.isclose(raw["accuracy"], cal["accuracy"])


def test_vector_fit_removes_bias():
    torch.manual_seed(0)
    true = torch.randn(3000, 4)
    y = torch.distributions.Categorical(logits=true).sample()
    biased = torch.log_softmax(true + torch.tensor([2.0, 0, 0, 0]), 1)
    c = fit(biased, y, "vector", l2=0.0)
    assert c["bias"][0] - sum(c["bias"][1:]) / 3 < -1.5
    assert metrics(apply(biased, c), y)["nll"] < metrics(biased, y)["nll"]


def test_rotation_maps_back(tiny):
    planner = Planner(AutoTokenizer.from_pretrained(tiny))
    plans = planner.plans("x", TOPIC, Profile(permutations=4))
    assert len(plans) == 4 and [p.perms[0][0] for p in plans] == [0, 1, 2, 3]
    shown = torch.tensor([10.0, 0.0, 0.0, 0.0])  # the first listed option wins in every read
    for p in plans:
        lp = canonical(p, [shown])[0]
        assert lp.argmax().item() == p.perms[0][0]
    avg = combine(plans, [[shown]] * 4)[0]
    assert torch.allclose(avg, avg[0].expand(4), atol=1e-6)  # pure position bias averages out


@pytest.mark.parametrize(("model", "layout"), [("tiny", "system"), ("small", "inline")])
def test_name_labels(model, layout, request):
    planner = Planner(AutoTokenizer.from_pretrained(request.getfixturevalue(model)))
    plan = planner.plan("x", TOPIC, label_style="names", layout=layout)
    assert plan.labels[0] == ["sports", "business", "science", "world"]
    multi = {"c": {"type": "choice", "criteria": {"science and tech": None, "world news": None}}}
    assert planner.plan("x", multi, label_style="names", layout=layout).labels[0] == ["A", "B"]  # letters fallback


def test_lora_merge_matches_unmerged(tiny):
    planner = Planner(AutoTokenizer.from_pretrained(tiny))
    plans = [planner.plan("Where is the station?", QUESTIONS)]
    runner = Runner(tiny, cache_bytes=0, merge_projections=False)
    lora.inject(runner.model, rank=4, alpha=8)
    torch.manual_seed(0)
    for m in runner.model.modules():
        if isinstance(m, lora.LoRALinear):
            m.lora_B.data.normal_(std=0.02)
    before = runner.read_many(plans)
    lora.merge(runner.model)
    after = runner.read_many(plans)
    for a, b in zip(before[0], after[0]):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_train_lora_lowers_loss(tiny, tmp_path):
    rows = [{"state": s, "questions": TOPIC, "labels": {"topic": t}} for s, t in [
        ("The striker scored twice in the final.", "sports"), ("Shares fell after weak earnings.", "business"),
        ("Researchers sequenced the genome of a fern.", "science"), ("The summit ended without a treaty.", "world"),
    ] * 2]
    runner = Runner(tiny, cache_bytes=0, merge_projections=False)
    planner = Planner(AutoTokenizer.from_pretrained(tiny))
    names = list(TOPIC["topic"]["criteria"])

    def dataset_loss():  # step losses are noisy (random rotations); judge the whole set
        plans = [planner.plan(r["state"], TOPIC) for r in rows]
        lps = [combine([p], [r])[0] for p, r in zip(plans, runner.read_many(plans))]
        return -sum(lp[names.index(r["labels"]["topic"])] for lp, r in zip(lps, rows)).item() / len(rows)

    before = dataset_loss()
    # fixed option order: a 135M model cannot learn shuffled letter labels in 12 steps
    train_lora(runner, planner, rows, epochs=6, lr=2e-3, rank=4, alpha=8, batch_size=4, log=lambda *_: None,
               rotate=False)
    assert dataset_loss() < before - 0.1
    lora.save(runner.model, tmp_path / "adapter")
    tuned = Runner(tiny, cache_bytes=0, adapter=tmp_path / "adapter")
    plan = planner.plan(rows[0]["state"], TOPIC)
    torch.testing.assert_close(tuned.read(plan)[0], runner.read(plan)[0], atol=1e-4, rtol=1e-4)


def test_profile_over_api(tiny):
    planner, runner = Planner(AutoTokenizer.from_pretrained(tiny)), Runner(tiny)
    hot = Profile(permutations=2, questions={"topic": {"temperature": 100.0}})
    app = create_app(ServerSettings(tiny, profiles={"flat": hot}), AsyncEngine(planner, runner))
    with TestClient(app) as c:
        body = {"state": "The striker scored twice.", "questions": TOPIC}
        raw = c.post("/v1/systemone", json={**body, "model": "dan-latest"}).json()
        flat = c.post("/v1/systemone", json={**body, "model": "dan-latest@flat"}).json()
        assert flat["model"] == "dan-latest@flat"
        assert flat["answers"]["topic"]["confidence"] < min(0.01, raw["answers"]["topic"]["confidence"])
        assert flat["usage"]["input_tokens"] > raw["usage"]["input_tokens"]  # two rotations read
        assert c.post("/v1/systemone", json={**body, "model": "dan-latest@nope"}).status_code == 400
        assert any(m["name"] == "dan-latest@flat" for m in c.get("/v1/models").json()["models"])


@pytest.mark.parametrize("style", ["letters", "names"])
def test_llm_profile_styles(tiny, style):
    from dan import LLM

    out = LLM(tiny).decide("The striker scored twice.", TOPIC, Profile(permutations=4, label_style=style))
    assert math.isclose(sum(out["topic"]["probabilities"].values()), 1.0, rel_tol=1e-6)
