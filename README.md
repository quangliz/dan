# dan

Fast, easy System One inference and serving: like vLLM, but for decisions.

dan turns an open-weight causal LM into a System One model. It returns typed, calibrated answers
(`noul` / `choice` / `score`, the same shapes as TypeSafe's Jev) read from label-token probabilities,
with one prefill per request and no generated tokens.

## Install

From source:

```bash
git clone https://github.com/quangliz/dan.git && cd dan
uv venv && uv pip install -e '.[dev]'     # editable, with test/lint tools; or: pip install -e .
```

As a dependency of another project:

```bash
uv add "dan @ git+https://github.com/quangliz/dan.git"      # or: pip install "git+https://github.com/quangliz/dan.git"
```

On NVIDIA GPUs, use a CUDA build of torch, and for Qwen3.5 also install the recurrent-layer kernels:

```bash
uv pip install flash-linear-attention   # without it, Qwen3.5 falls back to a slow pure-PyTorch path
```

## Quickstart

Python (`LLM` defaults to CPU / fp32; pass `device` and `dtype` for a GPU):

```python
import torch
from dan import LLM

llm = LLM("Qwen/Qwen3.5-4B", device="cuda", dtype=torch.bfloat16)
llm.decide({"ticket": "I was charged twice for order #1234, please fix this today!"}, {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"refund": "money back", "shipping": "where is my order", "other": None}},
    "urgent": {"type": "noul", "instructions": "Does the customer need action today?"},
    "anger": {"type": "score", "criteria": ["calm", "annoyed", "furious"]},
})
```

Server (Jev-compatible: TypeSafe's SDKs work with `TYPESAFE_BASE_URL=http://127.0.0.1:8000`):

```bash
dan serve Qwen/Qwen3.5-4B --port 8000        # CUDA + bf16 when available
curl localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "dan-latest", "state": "where is my parcel?",
  "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}}'
```

## Requests and answers

A request is a `state` (string or JSON) and named `questions`:

| type | `criteria` | answer |
|---|---|---|
| `noul` (yes/no) | optional `{"true": ..., "false": ...}` descriptions | `{"type": "noul", "noul": p_yes}` |
| `choice` | `{option: description or null, ...}`, up to 255 options | `{"type": "choice", "choice", "probabilities", "confidence"}` |
| `score` | an ordered list of 1–10 level descriptions | `{"type": "score", "score": expected level, "legend", "probabilities", "confidence"}` |

`confidence` is `1 − H(p) / ln K`: 1 when certain, 0 when uniform. A choice with one option or a
score with one level is answered without running the model. The server replies with `usage.input_tokens`
and `output_tokens: 0`. The Jev options `think`, `images` and `sequential` are not supported and are
refused with a 400.

## How it works

- **One prefill per request.** The state is a shared *trunk*. Each question (its instructions,
  lettered options and the assistant header) is its own *branch* after the trunk. Its answer is the
  softmax over its labels' next-token logits (`W_lm[label_ids] @ h`, never the full vocabulary).
- **Questions don't see each other.** In attention layers, every branch reads the trunk and itself
  only. On CUDA this is block-sparse: flash varlen calls for the trunk, for branch-vs-trunk and for
  branch-vs-itself, merged exactly by log-sum-exp. On CPU it's a masked reference path. In Qwen3.5's
  recurrent (Gated DeltaNet) layers, every branch starts from the trunk's final recurrent state.
- **Batched.** The server packs concurrent requests into one forward pass up to a token budget.
  Projections and MLPs run once over all packed tokens.
- **Prefix cache.** The state-independent start of the prompt is cached (keys/values, recurrent
  state and conv history) and reused across requests.

Supported architectures: Llama 2/3, Mistral, Qwen2/2.5/3 and SmolLM (`llama`, `mistral`, `qwen2`,
`qwen3`), and Qwen3.5 (`qwen3_5`, text only). Other model types are rejected at load time.

## Serving options

```bash
dan serve MODEL [--device cuda] [--dtype bfloat16] [--quantization fp8] [--adapter DIR] \
    [--profile NAME=profile.json ...] [--served-model-name dan-latest] [--api-key KEY] \
    [--max-batch-tokens 8192] [--max-wait-ms 2] [--max-queue 1024] [--prefix-cache-gb 2]
```

- `GET /v1/models`, `POST /v1/systemone`, `GET /health`, and `GET /metrics` (Prometheus text: requests,
  batches, latency, queue, prefix-cache hits). `jev-latest` / `jev-preview` are accepted as model names.
- `--quantization fp8` needs an sm89+ GPU (L4, H100, ...). It's faster, but probabilities are noisier
  than bf16 (answers near 50/50 can flip). Prefer bf16 when exact probabilities matter.
- `--api-key` (or `DAN_API_KEY`) requires `Authorization: Bearer <key>`. The server returns 529 when
  the queue is full.
- Tuning knobs (environment): `DAN_VARLEN=0` forces the reference attention path; `DAN_COMPILE=0`
  disables torch.compile of the fused ops; `DAN_COMPILE_MIN_ROWS` and `DAN_RECURRENT_MAX_TOKENS`
  (both default 512) set when fused ops compile and when short recurrent calls use the single-launch
  kernel.

## Calibration

Off-the-shelf instruct models are usually over-confident and favor some option positions. A
**profile**, fitted on labeled data, corrects that per task:

```bash
# labeled JSONL: {"state": ..., "questions": {...}, "labels": {"topic": "sports"}}
dan calibrate MODEL fit.jsonl --eval eval.jsonl --method vector --out topic.json
dan eval MODEL eval.jsonl --profile topic.json     # accuracy / ECE / Brier / NLL
dan serve MODEL --profile topic=topic.json         # request model "dan-latest@topic"
```

- Methods: `vector` (temperature + per-option bias, the recommended default), `temperature`, and
  `contextual` (bias from an "N/A" state; it can backfire when that state is itself informative).
- `--permutations k` reads k rotations of the option order and averages them against position bias
  (k× the compute). `--label-style names` uses option names as labels when each is one token.
- A profile also records the prompt layout. `inline` (the default) puts each question right before
  its answer, which small models need. `system` puts all questions in the system prompt.

For more accuracy, fine-tune a LoRA adapter with a proper scoring rule through the same read path.
Options are rotated randomly during training, so the adapter learns content, not positions:

```bash
dan train MODEL fit.jsonl --out adapter/ --loss log --epochs 2
dan serve MODEL --adapter adapter/     # merged into the weights: no serving overhead
```

Fit the profile for an adapter on rows the adapter was not trained on.

## Limitations

- Labels must be single tokens in the model's tokenizer. dan picks letters (or option names) that
  are, and refuses a question it can't label.
- No reasoning step: answers come from one read, so multi-step math and code questions are weaker
  than with generation.
- Long prompts are limited by GPU memory, not truncated.

## Development

```bash
pytest tests          # model tests skip unless their checkpoints are cached; GPU tests need CUDA
ruff check dan tests
```
