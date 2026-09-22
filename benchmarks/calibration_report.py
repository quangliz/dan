"""Accuracy and calibration of a model on the benchmark splits, across read
settings (label style, option rotations) and calibration methods.

    python benchmarks/make_datasets.py --out data
    python benchmarks/calibration_report.py Qwen/Qwen2.5-0.5B-Instruct --data data --out report.md

Each setting reads the fit and eval splits once; every method is then fitted
on the fit split's log-probabilities and scored on the eval split's.
"""
import argparse
import json
import os
import time

import torch

from dan import LLM
from dan.calibration import Profile, apply, fit, metrics
from dan.tuning import NULL_STATE, collect, load_rows

SETTINGS = [("letters", 1), ("names", 1), ("letters", 2), ("names", 2)]
METHODS = ["temperature", "vector", "contextual"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--data", default="data")
    p.add_argument("--sets", nargs="*", default=["ag_news", "sst2", "boolq"])
    p.add_argument("--adapter", default=None)
    p.add_argument("--layout", choices=("inline", "system"), default="inline")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="report.md")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    llm = LLM(args.model, adapter=args.adapter)
    lines = [f"# Calibration report: {args.model}" + (f" + {args.adapter}" if args.adapter else "")
             + f" ({args.layout} layout)", "",
             "| set | labels | rotations | method | acc | ECE | Brier | NLL | s/req |",
             "|---|---|---|---|---|---|---|---|---|"]
    results = []
    for name in args.sets:
        fit_rows = load_rows(os.path.join(args.data, f"{name}.fit.jsonl"))[: args.limit or None]
        eval_rows = load_rows(os.path.join(args.data, f"{name}.eval.jsonl"))[: args.limit or None]
        has_choice = any(q["type"] == "choice" for q in fit_rows[0]["questions"].values())
        for style, k in SETTINGS:
            if not has_choice and (style == "names" or k > 2):
                continue  # yes/no: labels are the same either way, and two rotations cover both orders
            profile = Profile(k, style, args.layout)
            t = time.time()
            train = collect(llm, fit_rows, profile)
            test = collect(llm, eval_rows, profile)
            per_req = (time.time() - t) / (len(fit_rows) + len(eval_rows))
            groups, out = llm.read([(NULL_STATE, fit_rows[0]["questions"])], profile)
            nulls = {s.key: lp for s, lp in zip(groups[0][0].specs, out[0])}
            for key, (lp, y) in test.items():
                rows = [("raw", metrics(lp, y))]
                for m in METHODS:
                    c = fit(*train[key], m, nulls.get(key))
                    rows.append((m, metrics(apply(lp, c), y)))
                for m, r in rows:
                    results.append({"set": name, "labels": style, "rotations": k, "method": m, **r})
                    lines.append(f"| {name} | {style} | {k} | {m} | {r['accuracy']:.3f} | {r['ece']:.3f} | "
                                 f"{r['brier']:.3f} | {r['nll']:.3f} | {per_req:.2f} |")
                    print(lines[-1], flush=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
