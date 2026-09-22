"""Calibration: profiles that adjust label log-probabilities, their fitting,
and the metrics that judge them.

A profile describes how a task is read and corrected:

    {"permutations": 2, "label_style": "letters", "layout": "inline",
     "questions": {"topic": {"temperature": 1.3, "bias": [0.1, -0.2, 0.0, 0.1]}},
     "default": {"temperature": 1.1}}

``permutations`` rotations of the options are read and averaged; then each
question's log-probabilities become ``logp / temperature + bias`` (bias only
when its length matches the option count). ``default`` covers questions the
profile does not name.
"""
import json
import math
from dataclasses import asdict, dataclass, field
from itertools import pairwise

import torch

METHODS = ("temperature", "vector", "contextual")


@dataclass
class Profile:
    permutations: int = 1
    label_style: str = "letters"
    layout: str = "inline"  # see prompt.py
    questions: dict = field(default_factory=dict)
    default: dict | None = None

    def adjust(self, key, logp):
        c = self.questions.get(key, self.default)
        if not c:
            return logp
        out = logp / c.get("temperature", 1.0)
        bias = c.get("bias")
        if bias is not None and len(bias) == out.shape[0]:
            out = out + torch.tensor(bias, dtype=out.dtype)
        return out

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def target_index(spec, label):
    """A dataset label -> canonical option index."""
    if spec.type == "noul":
        return 0 if label in (True, "yes", "true", 1) else 1
    if spec.type == "score":
        return int(label)
    names = [n for n, _ in spec.options]
    if label not in names:
        raise ValueError(f"label {label!r} is not an option of {spec.key!r}: {names}")
    return names.index(label)


# ---- metrics ---------------------------------------------------------------

def metrics(logp, target, bins=15):
    """logp: [N, K] log-probabilities; target: [N]. Accuracy, top-label ECE,
    multi-class Brier score and negative log-likelihood."""
    p = logp.exp()
    conf, pred = p.max(1)
    correct = (pred == target).float()
    ece = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for lo, hi in pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    onehot = torch.nn.functional.one_hot(target, p.shape[1]).float()
    return {"n": len(target), "accuracy": correct.mean().item(), "ece": ece,
            "brier": ((p - onehot) ** 2).sum(1).mean().item(),
            "nll": -logp.gather(1, target[:, None]).mean().item()}


# ---- fitting ---------------------------------------------------------------

def fit(logp, target, method="vector", null_logp=None, l2=1e-2):
    """Fit one question's correction on [N, K] log-probabilities. Returns
    {"temperature": T, "bias": [...]} (bias omitted for "temperature")."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    k = logp.shape[1]
    log_t = torch.zeros((), requires_grad=True)
    if method == "contextual":
        if null_logp is None:
            raise ValueError("contextual calibration needs the null-state log-probabilities")
        bias = -null_logp.detach().clone()
        params = [log_t]
    else:
        bias = torch.zeros(k, requires_grad=method == "vector")
        params = [log_t] + ([bias] if method == "vector" else [])
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def temp():
        # bounded: a model with no signal on a task would otherwise drive T to infinity
        return log_t.clamp(math.log(0.05), math.log(100.0)).exp()

    def closure():
        opt.zero_grad()
        # contextual: p ∝ p / p_null, then tempered: (logp - null) / T
        z = (logp + bias) / temp() if method == "contextual" else logp / temp() + bias
        loss = torch.nn.functional.cross_entropy(z, target)
        if method == "vector":
            loss = loss + l2 * (bias ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    out = {"temperature": float(temp().detach())}
    if method == "contextual":
        out["bias"] = [float(b) / out["temperature"] for b in bias]
    elif method == "vector":
        out["bias"] = [float(b) for b in bias.detach()]
    return out


def apply(logp, c):
    z = logp / c.get("temperature", 1.0)
    if c.get("bias") is not None:
        z = z + torch.tensor(c["bias"])
    return torch.log_softmax(z, dim=1)
