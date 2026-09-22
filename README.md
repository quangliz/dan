# dan

Fast, easy System One model inference and serving — like vLLM, but for decisions.

dan turns any open-weight causal LM into a System One model: typed, calibrated answers
(`noul` / `choice` / `score`, Jev-compatible) read from label-token probabilities in a
single prefill — no tokens generated.

## Quickstart

```bash
uv venv && uv pip install -e '.[dev]'
```

Offline:

```python
from dan import LLM

llm = LLM("Qwen/Qwen2.5-0.5B-Instruct")
llm.decide({"ticket": "I was charged twice!"}, {
    "intent": {"type": "choice", "criteria": {"refund": "money back", "shipping": None, "other": None}},
    "urgent": {"type": "noul", "instructions": "Does the customer need action today?"},
})
```

Server (Jev-compatible, so TypeSafe's SDKs work with `TYPESAFE_BASE_URL=http://127.0.0.1:8000`):

```bash
dan serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
curl localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "dan-latest", "state": "where is my parcel?",
  "questions": {"urgent": {"type": "noul"}}}'
```

## How it works

- The prompt is `[system: questions + labels] [user: state] [assistant header]`; each question is a
  short branch `q<n>:` whose next token is read over that question's single-token labels.
- All questions of all batched requests run in **one prefill**: projections/MLPs are batched over the
  packed tokens; attention runs per request under a tree mask (questions see the prompt, not each other).
- The static prompt prefix (instructions + schema) is KV-cached and reused across requests.
- Only label logits are computed (`W_lm[label_ids] @ h`), never the full vocabulary.

Supported architectures: Llama 2/3, Mistral, Qwen2/2.5/3, SmolLM.

## Calibration

Probabilities from an off-the-shelf instruct model are usually over-confident and biased toward
some option positions. dan fixes this per task with a **profile**, fitted on labeled data:

```bash
# labeled JSONL: {"state": ..., "questions": {...}, "labels": {"topic": "sports"}}
dan calibrate Qwen/Qwen2.5-0.5B-Instruct fit.jsonl --eval eval.jsonl --method vector \
    --permutations 2 --out topic.json          # prints accuracy / ECE / Brier / NLL, raw vs calibrated
dan eval Qwen/Qwen2.5-0.5B-Instruct eval.jsonl --profile topic.json
dan serve Qwen/Qwen2.5-0.5B-Instruct --profile topic=topic.json   # request model "dan-latest@topic"
```

- `--permutations k` reads k rotations of the option order and averages them (against position bias).
- `--label-style names` uses option names as labels when each is a single token (else letters).
- Methods: `temperature` (one T), `vector` (T + per-option bias), `contextual` (bias from a null state, then T).

For a stronger fix, fine-tune a LoRA adapter with a proper scoring rule through the same read path
(options are rotated randomly during training so the adapter learns content, not positions):

```bash
dan train Qwen/Qwen2.5-0.5B-Instruct fit.jsonl --out adapter/ --loss log --epochs 1
dan serve Qwen/Qwen2.5-0.5B-Instruct --adapter adapter/   # merged into the weights: no serving overhead
```

Benchmark data and a full report: `benchmarks/make_datasets.py`, `benchmarks/calibration_report.py`.
