import asyncio

import pytest
from fastapi.testclient import TestClient
from test_prompt_llm import QUESTIONS
from transformers import AutoTokenizer

from dan.engine import AsyncEngine
from dan.entrypoints.api_server import ServerSettings, create_app
from dan.prompt import Planner
from dan.runner import Runner


@pytest.fixture(scope="module")
def engine_parts(tiny):
    return Planner(AutoTokenizer.from_pretrained(tiny)), Runner(tiny), tiny


@pytest.fixture
def client(engine_parts):
    planner, runner, name = engine_parts
    app = create_app(ServerSettings(name, api_key="secret"), AsyncEngine(planner, runner))
    with TestClient(app, headers={"authorization": "Bearer secret"}) as c:
        yield c


def test_systemone(client):
    r = client.post("/v1/systemone", json={"model": "jev-latest", "state": {"msg": "hi?"}, "questions": QUESTIONS})
    assert r.status_code == 200, r.text
    body = r.json()
    assert list(body["answers"]) == list(QUESTIONS)
    assert body["answers"]["language"]["type"] == "choice"
    assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] == 0
    assert r.headers["x-request-id"].startswith("req_")


def test_errors(client):
    base = {"model": "dan-latest", "state": "x"}
    assert client.post("/v1/systemone", json={**base, "model": "nope", "questions": QUESTIONS}).json()["detail"][
        "error_type"] == "api_usage_error"
    assert client.post("/v1/systemone", json={**base, "questions": {"q": {"type": "bogus"}}}).status_code == 400
    assert client.post("/v1/systemone", json={**base, "questions": {}}).status_code == 422
    r = client.post("/v1/systemone", json={**base, "questions": {"s": {"type": "score", "criteria": list("abcdefghijk")}}})
    assert r.status_code == 400 and "score levels" in r.json()["detail"]
    assert client.post("/v1/systemone", json={**base, "questions": QUESTIONS, "think": 64}).status_code == 400
    assert client.post("/v1/systemone", json={**base, "questions": QUESTIONS},
                       headers={"authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/health").status_code == 200


def test_concurrent_requests_batch(engine_parts):
    planner, runner, _ = engine_parts
    states = [f"message number {i}, is this a question?" for i in range(12)]

    async def go():
        eng = AsyncEngine(planner, runner, max_wait_ms=20)
        out = await asyncio.gather(*[eng.decide(s, QUESTIONS) for s in states])
        await eng.stop()
        return eng, out

    eng, out = asyncio.run(go())
    assert eng.stats.requests == 12 and eng.stats.batches < 12
    alone = [runner.read(planner.plan(s, QUESTIONS)) for s in states[:3]]
    from dan.readout import answers
    for s, a, (got, _) in zip(states, alone, out):
        want = answers(planner.plan(s, QUESTIONS), a)
        assert got["language"]["choice"] == want["language"]["choice"]
        assert abs(got["is_question"]["noul"] - want["is_question"]["noul"]) < 1e-4
