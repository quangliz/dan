"""Label logits -> Jev answers."""
import math


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


def answers(plan, label_logits):
    """``label_logits``: per branch, the logits of that question's label tokens."""
    out = dict(plan.forced)
    for spec, logits in zip(plan.specs, label_logits):
        out[spec.key] = to_answer(spec, softmax([float(x) for x in logits]))
    return {k: out[k] for k in plan.order}
