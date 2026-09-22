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
