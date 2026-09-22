"""Jev-compatible request and answer types, and the internal question form.

Request shapes follow TypeSafe's published Jev API (as re-implemented by OpenJev,
Apache-2.0), so their SDKs can talk to dan unchanged.
"""
import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

JSONContent = str | dict[str, Any] | list[Any]
Described = str | dict[str, Any] | list[Any] | None

MAX_CHOICES = 255
MAX_SCORE_LEVELS = 10


class NoulCriteria(BaseModel):
    true: Described = None
    false: Described = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Described = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Described = None
    criteria: dict[str, Described]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Described = None
    criteria: list[JSONContent] = Field(min_length=1)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]
Questions = TypeAdapter(dict[str, Question])


class SchemaError(ValueError):
    """A request the model cannot answer as asked; surfaced as a 400."""

    def __init__(self, msg, loc=("body",)):
        super().__init__(msg)
        self.loc = list(loc)


def text_of(value):
    """Descriptions and instructions may be strings, objects or arrays."""
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


@dataclass
class Spec:
    """One question as the model sees it. ``options`` are (name, description)
    pairs in answer order; labels are assigned later, per tokenizer."""

    key: str
    type: str
    instructions: str
    options: list[tuple[str, str]]
    legend: list | None = None


def parse_questions(questions):
    """Jev questions (dicts or pydantic models) -> (specs to read, forced answers).
    A choice with one option, or a score with one level, has only one possible
    answer, so it is answered here rather than read."""
    parsed = Questions.validate_python(
        {k: q.model_dump() if isinstance(q, BaseModel) else q for k, q in questions.items()})
    specs, forced = [], {}
    for key, q in parsed.items():
        loc = ("body", "questions", key, "criteria")
        if q.type == "noul":
            crit = q.criteria or NoulCriteria()
            options = [("yes", text_of(crit.true)), ("no", text_of(crit.false))]
            legend = None
        elif q.type == "choice":
            if not q.criteria:
                raise SchemaError(f"Choice question must have at least one choice: {key}", loc)
            if len(q.criteria) > MAX_CHOICES:
                raise SchemaError(f"Too many choices. Must have at most {MAX_CHOICES} choices.", loc)
            if len(q.criteria) == 1:
                only = next(iter(q.criteria))
                forced[key] = {"type": "choice", "choice": only, "probabilities": {only: 1.0}, "confidence": 1.0}
                continue
            options = [(name, text_of(desc)) for name, desc in q.criteria.items()]
            legend = None
        else:
            if len(q.criteria) > MAX_SCORE_LEVELS:
                raise SchemaError(f"Too many score levels. Must have at most {MAX_SCORE_LEVELS} levels.", loc)
            if len(q.criteria) == 1:
                forced[key] = {"type": "score", "score": 0.0, "legend": {"0": q.criteria[0]},
                               "probabilities": {"0": 1.0}, "confidence": 1.0}
                continue
            options = [(str(i), text_of(c)) for i, c in enumerate(q.criteria)]
            legend = list(q.criteria)
        specs.append(Spec(key, q.type, text_of(q.instructions), options, legend))
    return specs, forced
