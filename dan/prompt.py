"""Turn a state and Jev questions into a read plan: one shared prompt prefix
(the trunk) and, per question, a suffix (a branch) whose next-token
distribution over that question's label tokens is the answer. The runner reads
all branches in one pass under a tree mask, so questions never see each other.

Two layouts:

- ``inline`` (default): trunk = [user: state ...], branch = "... question and
  options. Reply with the label only." + end of turn + assistant header; the
  label is the first token of the reply. The question sits right before the
  answer, which small models need (Qwen2.5-0.5B on AG News: 70% vs 32%).
- ``system``: trunk = [system: all questions] [user: state] [assistant header],
  branch = "q<n>:". The schema comes first, so its keys and values are cached
  across requests, but small models follow it poorly.

Label selection and slot checks are adapted from OpenJev (Apache-2.0).
"""
import json
import string
from dataclasses import dataclass, field, replace

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
    perms: list[list[int]] = field(default_factory=list)  # per question: option index shown at each position


ROTATABLE = ("choice", "noul")  # score levels are ordered; their order is meaning, not bias


def rotation_perm(spec, rotation, rotations):
    """Rotation ``rotation`` of ``rotations``, spread evenly over the options."""
    k = len(spec.options)
    r = (rotation * k) // rotations if spec.type in ROTATABLE else 0
    return [(i + r) % k for i in range(k)]


class Planner:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self._slots = {}
        self._static = {}
        self._supports_system = None

    def enc(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def plans(self, state, questions, profile=None):
        """The reads one request takes under ``profile``: one per option
        rotation (at most as many as the largest option list)."""
        style = profile.label_style if profile else "letters"
        layout = getattr(profile, "layout", "inline") if profile else "inline"
        n = profile.permutations if profile else 1
        first = self.plan(state, questions, 0, n, style, layout)
        n = min(n, max([len(s.options) for s in first.specs if s.type in ROTATABLE], default=1))
        return [first] + [self.plan(state, questions, r, n, style, layout) for r in range(1, n)]

    def plan(self, state, questions, rotation=0, rotations=1, label_style="letters", layout="inline"):
        """``rotation`` of ``rotations`` shifts the listed order of choice and
        yes/no options, so reads can be averaged against position bias.
        ``label_style``: "letters" (A, B, ...) or "names" (a choice's own option
        names when each is one distinct token, else letters).
        ``layout``: "inline" or "system" (see the module docstring)."""
        if label_style not in ("letters", "names"):
            raise ValueError(f"unknown label style {label_style!r}")
        if layout not in ("inline", "system"):
            raise ValueError(f"unknown layout {layout!r}")
        specs, forced = parse_questions(questions)
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        perms = [rotation_perm(s, rotation, rotations) for s in specs]
        shown = [replace(s, options=[s.options[i] for i in perm]) for s, perm in zip(specs, perms)]
        if not specs:
            return ReadPlan([], specs, [], [], forced, list(questions), 0, perms)
        system = layout == "system"
        # Labels depend only on the assistant header, which is fixed per model:
        # choose them against a probe prompt, then check them against the real one.
        tail = self.prefix_ids("" if system else None, "")[-TAIL:]
        leads = [f"q{n + 1}: " if system else "" for n in range(len(specs))]
        labels = [self.choose_labels(s, tail, label_style, lead) for s, lead in zip(shown, leads)]
        if system:
            text = self.system_text(shown, labels)
            prefix = self.prefix_ids(text, state_text)
            branches = [self.branch(labs, prefix[-TAIL:], s.key, lead)
                        for s, labs, lead in zip(specs, labels, leads)]
            static = self.common(prefix, self.cached_ids(text, ""))
        else:
            fulls = [self.prefix_ids(None, f"{state_text}\n\n{self.question_text(s, labs)}")
                     for s, labs in zip(shown, labels)]
            probe = self.prefix_ids(None, f"{state_text}\n\n#")  # where any question would start
            n = min(self.common(f, probe) for f in fulls)
            prefix = fulls[0][:n]
            branches = []
            for s, labs, full in zip(specs, labels, fulls):
                b = self.branch(labs, full[-TAIL:], s.key, "")
                branches.append(Branch(full[n:], b.label_ids))
            static = self.common(prefix, self.cached_ids(None, ""))
        return ReadPlan(prefix, specs, labels, branches, forced, list(questions),
                        min(static, len(prefix) - 1), perms)  # keep one trunk token computed fresh

    def cached_ids(self, system, user):
        key = (system, user)
        hit = self._static.get(key)
        if hit is None:
            hit = self.prefix_ids(system, user)
            if len(self._static) > 1024:
                self._static.clear()
            self._static[key] = hit
        return hit

    @staticmethod
    def common(a, b):
        """Length of the common prefix of two token lists."""
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def choose_labels(self, spec, tail, label_style="letters", lead=""):
        """Labels for ``spec``'s options in the order they are shown, each one
        token after ``tail`` + ``lead``."""
        k = len(spec.options)
        if spec.type == "noul":
            names = [name for name, _ in spec.options]
            sets = [[{"yes": y, "no": no}[x] for x in names] for y, no in NOUL_LABELS]
        elif spec.type == "score":
            sets = [DIGITS[:k], LETTERS[:k]]
        else:
            if label_style == "names":
                names = [name for name, _ in spec.options]
                if self.pick(names, k, tail, lead) is not None:
                    return names
            got = self.pick(CHOICE_POOL, k, tail, lead)
            if got is None:
                raise SchemaError(f"this model's tokenizer has too few single-token labels for {k} choices",
                                  ("body", "questions", spec.key, "criteria"))
            return got
        for cands in sets:
            if self.pick(cands, k, tail, lead) is not None:
                return list(cands)
        raise SchemaError("no single-token label set fits this model's tokenizer", ("body", "questions", spec.key))

    def pick(self, cands, k, tail, lead):
        """The first k candidates that each add exactly one distinct token after
        the shared ``lead`` (e.g. "q1: "), or None."""
        suffix, out, seen = None, [], set()
        for c in cands:
            ids = self.continuation(tail, lead + c)
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

    def branch(self, labels, tail, key, lead):
        """The tokens ``lead`` adds after ``tail`` and each label's single token."""
        rows = [self.continuation(tail, lead + lab) for lab in labels]
        suffix = rows[0][:-1]
        if (lead and not suffix) or any(len(r) != len(suffix) + 1 or r[:-1] != suffix for r in rows):
            raise SchemaError(f"question {key!r}: labels do not share one read position", ("body", "questions", key))
        return Branch(suffix, [r[-1] for r in rows])

    @staticmethod
    def option_lines(spec, labels):
        s = ""
        for (name, desc), lab in zip(spec.options, labels):
            if spec.type == "choice" and lab != name:
                s += f"  {lab}: {name} ({desc})\n" if desc else f"  {lab}: {name}\n"
            else:
                s += f"  {lab}: {desc}\n" if desc else f"  {lab}\n"
        return s

    def question_text(self, spec, labels):
        """One question as the inline layout asks it, after the state."""
        q = spec.instructions or "Answer about the state above."
        return f"{q}\nChoose one:\n{self.option_lines(spec, labels)}\nReply with the label only."

    def system_text(self, specs, labels):
        s = ("Answer a fixed set of questions about the state the user provides. "
             "Each question lists its allowed answers; reply with exactly one label per question.\n")
        for n, (spec, labs) in enumerate(zip(specs, labels), 1):
            s += f"\nQuestion q{n}: {spec.instructions or 'Answer about the state.'}\n"
            s += self.option_lines(spec, labs)
        return s + '\nAnswer each question on its own line formatted as "id: label". Lines may come in any order.'

    def prefix_ids(self, system, user):
        """Chat-template token ids ending with the assistant header; no system
        message when ``system`` is None."""
        if system is None:
            msgs = [{"role": "user", "content": user}]
        elif self._supports_system is False:
            msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
        else:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
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
