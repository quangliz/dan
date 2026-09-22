"""Label logits -> Jev answers.

A request may be read under several option rotations (see
``Planner.plan``). Each read's label logits are put back in canonical option
order and turned into log-probabilities; the reads are averaged in log space,
a calibration profile (if any) adjusts the result, and that becomes the answer.
"""
import math

import torch


def softmax(xs):
    mx = max(xs)
    ex = [math.exp(x - mx) for x in xs]
    z = sum(ex)
    return [e / z for e in ex]


def confidence(p):
    """How peaked a distribution is: 1 - H(p)/ln(K). 1 is certain, 0 uniform."""
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(len(p))))


def to_answer(spec, p):
    if spec.type == "noul":
        return {"type": "noul", "noul": p[0]}
    if spec.type == "choice":
        top = max(range(len(p)), key=p.__getitem__)
        return {"type": "choice", "choice": spec.options[top][0],
                "probabilities": {o[0]: v for o, v in zip(spec.options, p)}, "confidence": confidence(p)}
    return {"type": "score", "score": sum(i * v for i, v in enumerate(p)),
            "legend": {str(i): lev for i, lev in enumerate(spec.legend)},
            "probabilities": {str(i): v for i, v in enumerate(p)}, "confidence": confidence(p)}


def canonical(plan, label_logits):
    """Per question, log-probabilities in canonical option order."""
    out = []
    for perm, lg in zip(plan.perms or [None] * len(plan.specs), label_logits):
        lg = torch.as_tensor(lg).float()
        if perm is not None:  # shown position i holds option perm[i]; invert (differentiably)
            inv = [0] * len(perm)
            for i, o in enumerate(perm):
                inv[o] = i
            lg = lg[torch.tensor(inv, device=lg.device)]
        out.append(torch.log_softmax(lg, dim=0))
    return out


def combine(plans, reads):
    """Average each question's log-probabilities over reads of the same
    request under different rotations."""
    per = [canonical(p, r) for p, r in zip(plans, reads)]
    return [torch.stack(qs).mean(0) for qs in zip(*per)] if per else []


def answers(plan, label_logits, profile=None, logits_canonical=False):
    """``label_logits``: per question, the logits of its label tokens (as read,
    unless ``logits_canonical``)."""
    out = dict(plan.forced)
    logits = label_logits if logits_canonical else canonical(plan, label_logits)
    for spec, lg in zip(plan.specs, logits):
        if profile is not None:
            lg = profile.adjust(spec.key, lg)
        out[spec.key] = to_answer(spec, softmax([float(x) for x in lg]))
    return {k: out[k] for k in plan.order}
