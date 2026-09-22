"""Evaluate, calibrate and fine-tune on labeled data.

Data is JSONL, one request per line with its true answers:

    {"state": ..., "questions": {...}, "labels": {"topic": "sports", "urgent": true}}

``labels`` gives an option name (choice), true/false (noul) or a level index
(score); questions without a label are ignored.
"""
import json
import random
import time
from collections import defaultdict

import torch

from .calibration import METHODS, Profile, apply, fit, metrics, target_index
from .readout import combine

NULL_STATE = "N/A"


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def collect(llm, rows, profile=None, batch_size=16):
    """Per question id: stacked [N, K] log-probabilities (rotations averaged,
    no calibration) and [N] target indices."""
    logps, targets = defaultdict(list), defaultdict(list)
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        groups, out = llm.read([(r["state"], r["questions"]) for r in chunk], profile)
        for row, plans, lps in zip(chunk, groups, out):
            for spec, lp in zip(plans[0].specs, lps):
                if spec.key in row.get("labels", {}):
                    logps[spec.key].append(lp)
                    targets[spec.key].append(target_index(spec, row["labels"][spec.key]))
    return {k: (torch.stack(logps[k]), torch.tensor(targets[k])) for k in logps}


def evaluate(llm, rows, profile=None, batch_size=16):
    """Per question id: metrics of the (calibrated, when ``profile`` has
    corrections) answers."""
    data = collect(llm, rows, profile, batch_size)
    out = {}
    for key, (lp, y) in data.items():
        c = profile.questions.get(key, profile.default) if profile else None
        out[key] = metrics(apply(lp, c) if c else lp, y)
    return out


def calibrate(llm, fit_rows, eval_rows=None, method="vector", permutations=1, label_style="letters", batch_size=16):
    """Fit a profile on ``fit_rows``; report raw vs calibrated metrics on
    ``eval_rows`` (if given). Returns (profile, report)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    profile = Profile(permutations, label_style)
    train = collect(llm, fit_rows, profile, batch_size)
    nulls = {}
    if method == "contextual":
        seen = {}
        for r in fit_rows:
            for k in r["questions"]:
                seen.setdefault(k, r["questions"])
        for key, qs in seen.items():
            groups, out = llm.read([(NULL_STATE, qs)], profile)
            nulls.update({s.key: lp for s, lp in zip(groups[0][0].specs, out[0]) if s.key == key})
    for key, (lp, y) in train.items():
        profile.questions[key] = fit(lp, y, method, nulls.get(key))
    report = {}
    if eval_rows:
        test = collect(llm, eval_rows, profile, batch_size)
        for key, (lp, y) in test.items():
            c = profile.questions.get(key)
            report[key] = {"raw": metrics(lp, y), "calibrated": metrics(apply(lp, c), y) if c else None}
    return profile, report


def train_lora(runner, planner, rows, epochs=1, lr=2e-4, rank=16, alpha=32, batch_size=8, loss="log",
               label_style="letters", rotate=True, seed=0, log=print):
    """Fine-tune ``runner.model`` with LoRA on proper-scoring-rule losses over
    the label distributions, through the same read path used for serving.
    ``rotate`` reads each example under a random option rotation, so the
    adapter learns content, not positions. Returns per-step losses."""
    from .lora import inject

    if loss not in ("log", "brier"):
        raise ValueError("loss is 'log' (cross-entropy) or 'brier'")
    rng = random.Random(seed)
    params = inject(runner.model, rank, alpha)
    runner.model.train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = max(1, (len(rows) + batch_size - 1) // batch_size) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
    losses, t0 = [], time.time()
    for epoch in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = [rows[j] for j in order[i:i + batch_size]]
            plans = []
            for r in batch:
                rot = rng.randrange(1 << 16) if rotate else 0
                plans.append(planner.plan(r["state"], r["questions"], rot % 64, 64, label_style))
            reads = runner.forward(plans, use_cache=False)
            terms = []
            for r, plan, read in zip(batch, plans, reads):
                for spec, lp in zip(plan.specs, combine([plan], [read])):
                    if spec.key not in r.get("labels", {}):
                        continue
                    y = target_index(spec, r["labels"][spec.key])
                    if loss == "log":
                        terms.append(-lp[y])
                    else:
                        onehot = torch.nn.functional.one_hot(torch.tensor(y), lp.shape[0]).float()
                        terms.append(((lp.exp() - onehot) ** 2).sum())
            if not terms:
                continue
            value = torch.stack(terms).mean()
            opt.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(value.item())
            if len(losses) % 10 == 0 or len(losses) == steps:
                log(f"epoch {epoch} step {len(losses)}/{steps} loss {sum(losses[-10:]) / len(losses[-10:]):.4f} "
                    f"({time.time() - t0:.0f}s)")
    runner.model.eval()
    return losses
