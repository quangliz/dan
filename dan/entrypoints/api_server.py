"""Jev-compatible HTTP API: POST /v1/systemone, GET /v1/models, /health, /metrics.

Request, response and error shapes follow TypeSafe's published Jev API (as
re-implemented by OpenJev, Apache-2.0), so their SDKs work against dan by
pointing TYPESAFE_BASE_URL at it.
"""
import hmac
import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..engine import AsyncEngine, Overloaded
from ..schema import JSONContent, Question, SchemaError

log = logging.getLogger("dan")

# Accepted so TypeSafe's SDKs work unchanged (their default model is jev-latest).
JEV_ALIASES = ("jev-latest", "jev-preview")


@dataclass
class ServerSettings:
    model: str
    served_name: str = "dan-latest"
    api_key: str = ""
    profiles: dict = field(default_factory=dict)  # name -> calibration.Profile, requested as "<model>@<name>"


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(min_length=1)
    # Jev/OpenJev options dan does not implement yet; rejected when set, never ignored.
    images: list | None = None
    think: int | None = None
    sequential: bool | None = None


def error(status, error_type, message, headers=None):
    return JSONResponse({"detail": {"error_type": error_type, "message": message}}, status_code=status, headers=headers)


TRIM_DEPTH, TRIM_ITEMS, TRIM_CHARS = 4, 20, 500


def trim(value, depth=0):
    """A value safe to echo in an error: deep or long parts become a placeholder."""
    if isinstance(value, str):
        return value if len(value) <= TRIM_CHARS else value[:TRIM_CHARS] + "..."
    if isinstance(value, dict | list):
        if depth >= TRIM_DEPTH:
            return "..."
        if isinstance(value, dict):
            return {str(k): trim(v, depth + 1) for k, v in list(value.items())[:TRIM_ITEMS]}
        return [trim(v, depth + 1) for v in value[:TRIM_ITEMS]]
    if isinstance(value, int | float | bool) or value is None:
        return value
    return str(value)[:TRIM_CHARS]


def create_app(settings, engine: AsyncEngine):
    names = {settings.served_name, *JEV_ALIASES}

    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        yield
        await engine.stop()

    app = FastAPI(title="dan", version=__version__, lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_body(request: Request, exc: RequestValidationError):
        if any(e.get("type") == "union_tag_invalid" for e in exc.errors()):
            return error(400, "api_usage_error", "Invalid request.")  # unknown question type, as Jev answers it
        errors = [{"type": e.get("type"), "loc": list(e.get("loc", ())), "msg": e.get("msg"),
                   "input": trim(e.get("input"))} for e in exc.errors()]
        log.warning("422 %s", "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['type']}" for e in errors))
        return JSONResponse({"detail": errors}, status_code=422)

    @app.middleware("http")
    async def request_id_and_auth(request: Request, call_next):
        rid = "req_" + secrets.token_hex(16)
        if request.url.path.startswith("/v1/") and settings.api_key:
            token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not token:
                resp = error(403, "authentication_error", "Must supply an API key! Check your request and try again.")
            elif not hmac.compare_digest(token, settings.api_key):
                resp = error(401, "authentication_error", "Cannot authenticate with the server. Please check your API key and try again.")
            else:
                resp = None
            if resp is not None:
                resp.headers["x-request-id"] = rid
                return resp
        response = await call_next(request)
        response.headers["x-typesafe-request-id"] = rid
        response.headers["x-request-id"] = rid
        return response

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        desc = f"dan {__version__} serving {settings.model}"
        return {"models": [{"name": n, "description": desc} for n in [settings.served_name, *JEV_ALIASES]]
                + [{"name": f"{settings.served_name}@{p}", "description": f"{desc}, calibration profile {p!r}"}
                   for p in settings.profiles]}

    @app.post("/v1/systemone")
    async def systemone(req: SystemOneRequest):
        base, _, profile_name = req.model.partition("@")
        if base not in names or (profile_name and profile_name not in settings.profiles):
            return error(400, "api_usage_error", f"Unknown model: {req.model}")
        profile = settings.profiles.get(profile_name) if profile_name else None
        for opt in ("images", "think", "sequential"):
            if getattr(req, opt):
                return JSONResponse({"detail": f"{opt} is not supported by dan yet"}, status_code=400)
        try:
            answers, tokens = await engine.decide(req.state, {k: q.model_dump() for k, q in req.questions.items()},
                                                  profile)
        except SchemaError as e:
            return JSONResponse({"detail": str(e)}, status_code=400)
        except Overloaded as e:
            return error(529, "overloaded_error", str(e), {"retry-after": "1"})
        return {"model": settings.served_name + (f"@{profile_name}" if profile_name else ""),
                "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        s, cache = engine.stats, getattr(engine.runner, "cache", None)
        rows = [("dan_requests_total", s.requests), ("dan_request_errors_total", s.errors),
                ("dan_batches_total", s.batches), ("dan_prompt_tokens_total", s.tokens),
                ("dan_busy_seconds_total", round(s.busy_s, 6)), ("dan_request_latency_seconds_sum", round(s.latency_s, 6)),
                ("dan_queue_size", engine.queue.qsize())]
        if cache is not None:
            rows += [("dan_prefix_cache_hits_total", cache.hits), ("dan_prefix_cache_misses_total", cache.misses),
                     ("dan_prefix_cache_bytes", cache.used)]
        return "".join(f"{k} {v}\n" for k, v in rows)

    return app
