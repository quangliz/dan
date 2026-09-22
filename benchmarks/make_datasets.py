"""Write public classification sets as dan calibration JSONL.

Each row: {"state": ..., "questions": {...}, "labels": {question id: answer}},
where an answer is an option name (choice), true/false (noul) or a level
index (score).

    python benchmarks/make_datasets.py --out data --n 800

writes <set>.jsonl plus a half/half split: <set>.fit.jsonl (to calibrate or
train on) and <set>.eval.jsonl (held out).
"""
import argparse
import json
import os
import random

AG_NEWS = {"world": "world news, politics, international affairs", "sports": "sports",
           "business": "business, economy, companies, markets", "science": "science and technology"}


def ag_news(rows):
    names = list(AG_NEWS)
    q = {"topic": {"type": "choice", "instructions": "What is the topic of the news article?", "criteria": AG_NEWS}}
    for r in rows:
        yield {"state": r["text"], "questions": q, "labels": {"topic": names[r["label"]]}}


def sst2(rows):
    q = {"positive": {"type": "noul", "instructions": "Is the sentiment of the movie review positive?"}}
    for r in rows:
        yield {"state": r["sentence"].strip(), "questions": q, "labels": {"positive": bool(r["label"])}}


def boolq(rows):
    q = {"answer": {"type": "noul", "instructions": "Using the passage, is the answer to the question yes?"}}
    for r in rows:
        yield {"state": {"passage": r["passage"], "question": r["question"] + "?"}, "questions": q,
               "labels": {"answer": bool(r["answer"])}}


SETS = {"ag_news": ("fancyzhx/ag_news", "test", ag_news),
        "sst2": ("stanfordnlp/sst2", "validation", sst2),
        "boolq": ("google/boolq", "validation", boolq)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--sets", nargs="*", default=list(SETS))
    args = p.parse_args()
    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    for name in args.sets:
        repo, split, convert = SETS[name]
        ds = load_dataset(repo, split=split)
        idx = random.Random(0).sample(range(len(ds)), min(args.n, len(ds)))
        rows = [json.dumps(row, ensure_ascii=False) + "\n" for row in convert(ds[i] for i in idx)]
        half = len(rows) // 2
        for suffix, part in (("", rows), (".fit", rows[:half]), (".eval", rows[half:])):
            path = os.path.join(args.out, f"{name}{suffix}.jsonl")
            with open(path, "w") as f:
                f.writelines(part)
            print(path, len(part))


if __name__ == "__main__":
    main()
