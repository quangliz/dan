"""Turn a state and Jev questions into a read plan: one shared prompt prefix and,
per question, a short suffix whose next-token distribution over that question's
label tokens is the answer.

The prompt is [system: instructions + question list] [user: state] [assistant
header]. Every question's suffix is "q<n>:" right after the assistant header,
so each question is read as its own branch off the shared prefix; the runner
decides whether branches run one by one or together under a tree mask.
Label selection and slot checks are adapted from OpenJev (Apache-2.0).
"""
import json
import string
from dataclasses import dataclass, field

from .schema import SchemaError, Spec, parse_questions

TAIL = 8  # prefix tokens re-tokenized with each suffix, so merges at the seam are seen

NOUL_LABELS = [["yes", "no"], ["Yes", "No"], ["Y", "N"], ["A", "B"]]
LETTERS = list(string.ascii_uppercase) + list(string.ascii_lowercase)
CHOICE_POOL = LETTERS + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
DIGITS = [str(i) for i in range(10)]


@dataclass
class Branch:
    suffix: list[int]  # tokens after the shared prefix; the last one is the read position
    label_ids: list[int]  # one token per option, in option order


@dataclass
class ReadPlan:
    prefix: list[int]
    specs: list[Spec]
    labels: list[list[str]]
    branches: list[Branch]
    forced: dict = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # question keys as the request listed them
    static: int = 0  # leading prefix tokens that do not depend on the state (cacheable)


class Planner:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self._slots = {}
        self._static = {}
        self._supports_system = None

    def enc(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def plan(self, state, questions):
        specs, forced = parse_questions(questions)
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        # Labels depend only on the assistant header, which is fixed per model:
        # choose them against a probe prompt, then check them against the real one.
        probe = self.prefix_ids("", "")
        labels = [self.choose_labels(n + 1, s, probe[-TAIL:]) for n, s in enumerate(specs)]
        system = self.system_text(specs, labels)
        prefix = self.prefix_ids(system, state_text) if specs else []
        branches = [self.branch(n + 1, labs, prefix[-TAIL:], s.key) for n, (s, labs) in enumerate(zip(specs, labels))]
        static = self.static_len(system, prefix) if specs else 0
        return ReadPlan(prefix, specs, labels, branches, forced, list(questions), static)

    def static_len(self, system, prefix):
        """How many leading prefix tokens are the same for any state: the
        common prefix with an empty-state render of the same system text."""
        probe = self._static.get(system)
        if probe is None:
            probe = self.prefix_ids(system, "")
            if len(self._static) > 1024:
                self._static.clear()
            self._static[system] = probe
        n = 0
        for a, b in zip(prefix, probe):
            if a != b:
                break
            n += 1
        return min(n, len(prefix) - 1)  # keep at least one token computed fresh

    def choose_labels(self, n, spec, tail):
        k = len(spec.options)
        if spec.type == "noul":
            sets = NOUL_LABELS
        elif spec.type == "score":
            sets = [DIGITS[:k], LETTERS[:k]]
        else:
            got = self.pick(n, CHOICE_POOL, k, tail)
            if got is None:
                raise SchemaError(f"this model's tokenizer has too few single-token labels for {k} choices",
                                  ("body", "questions", spec.key, "criteria"))
            return got
        for cands in sets:
            if self.pick(n, cands, k, tail) is not None:
                return list(cands)
        raise SchemaError("no single-token label set fits this model's tokenizer", ("body", "questions", spec.key))

    def pick(self, n, cands, k, tail):
        """The first k candidates that each add exactly one distinct token after
        the shared suffix "q<n>:", or None."""
        suffix, out, seen = None, [], set()
        for c in cands:
            ids = self.continuation(tail, f"q{n}: {c}")
            if suffix is None:
                suffix = ids[:-1]
            if len(ids) == len(suffix) + 1 and ids[:-1] == suffix and ids[-1] not in seen:
                seen.add(ids[-1])
                out.append(c)
                if len(out) == k:
                    return out
        return None

    def continuation(self, tail, text):
        """Token ids ``text`` becomes when it directly follows ``tail``."""
        key = (tuple(tail), text)
        hit = self._slots.get(key)
        if hit is None:
            head = self.tok.decode(tail)
            ids = self.enc(head + text)
            if ids[: len(tail)] != list(tail):
                ids = list(tail) + self.enc(text)  # the tail does not round-trip; tokenize apart
            hit = ids[len(tail):]
            if len(self._slots) > 65536:
                self._slots.clear()
            self._slots[key] = hit
        return hit

    def branch(self, n, labels, tail, key):
        rows = [self.continuation(tail, f"q{n}: {lab}") for lab in labels]
        suffix = rows[0][:-1]
        if not suffix or any(len(r) != len(suffix) + 1 or r[:-1] != suffix for r in rows):
            raise SchemaError(f"question {key!r}: labels do not share one read position", ("body", "questions", key))
        return Branch(suffix, [r[-1] for r in rows])

    def system_text(self, specs, labels):
        s = ("Answer a fixed set of questions about the state the user provides. "
             "Each question lists its allowed answers; reply with exactly one label per question.\n")
        for n, (spec, labs) in enumerate(zip(specs, labels), 1):
            s += f"\nQuestion q{n}: {spec.instructions or 'Answer about the state.'}\n"
            for (name, desc), lab in zip(spec.options, labs):
                if spec.type == "choice":
                    s += f"  {lab}: {name} ({desc})\n" if desc else f"  {lab}: {name}\n"
                else:
                    s += f"  {lab}: {desc}\n" if desc else f"  {lab}\n"
        return s + '\nAnswer each question on its own line formatted as "id: label". Lines may come in any order.'

    def prefix_ids(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self._supports_system is False:
            msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
        try:
            out = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False)
            self._supports_system = self._supports_system if self._supports_system is not None else True
        except Exception:
            if self._supports_system is not None:
                raise
            self._supports_system = False  # e.g. Gemma 2: no system role
            return self.prefix_ids(system, user)
        ids = out["input_ids"] if hasattr(out, "keys") else out
        return [int(t) for t in ids]
