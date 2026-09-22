# Benchmark results

Model: `Qwen/Qwen2.5-0.5B-Instruct`, fp32 on CPU (14 cores). Data from `benchmarks/make_datasets.py`:
200 labeled rows to fit on and 200 held-out rows to evaluate, per set. Accuracy / ECE (15 bins,
top label) / NLL on the held-out rows. Full tables: `qwen2.5-0.5b-inline.md` (the default layout)
and `qwen2.5-0.5b-system.md`.

## Prompt layout

Where the question sits matters most for a small model. `system` puts every question in the system
prompt before the state; `inline` (the default) puts each question and its options right before the
answer, in that question's own tree branch.

| set | system layout, raw | inline layout, raw |
|---|---|---|
| AG News (4 topics) | 27.5% / 0.157 | **80.0% / 0.159** |
| SST-2 (yes/no) | 79.5% / 0.099 | **88.5% / 0.120** |
| BoolQ (yes/no) | 66.5% / 0.097 | 65.5% / 0.167 |

## Calibration and LoRA (inline layout, letter labels, one read)

| set | raw | vector profile | LoRA | LoRA + vector profile |
|---|---|---|---|---|
| AG News | 80.0% / 0.159 / 0.773 | 84.0% / **0.049** / 0.474 | 87.0% / 0.110 / 0.651 | **87.5%** / 0.074 / **0.429** |
| SST-2 | 88.5% / 0.120 / 0.311 | **89.5% / 0.053 / 0.285** | – | – |
| BoolQ | 65.5% / 0.167 / 0.751 | 67.0% / **0.059** / 0.586 | 72.0% / 0.166 / 0.705 | **74.0%** / 0.095 / **0.558** |

LoRA: rank 16, 2 epochs over 200 rows, lr 5e-4, log loss, random option rotations (about 7–10 min on
CPU). Its profile was fitted on 200 *other* fit rows: fitting on the rows the adapter trained on
would learn its training-set overconfidence.

## Takeaways

- Use the inline layout (default). The system layout's cacheable schema prefix is not worth its
  accuracy loss on small models.
- A `vector` profile is the cheapest large win: ECE drops 2–3x at no serving cost.
- LoRA buys accuracy (+7 points on AG News and BoolQ); follow it with a profile for calibration.
- Two option rotations help yes/no calibration (SST-2 ECE 0.120 → 0.069) at 2x cost; on AG News they
  do not help once the inline layout is used.
- Letter labels beat option-name labels before calibration (AG News 80.0% vs 75.5%).
- `contextual` calibration is unreliable: on SST-2 the "N/A" null state reads as negative and the
  correction halves accuracy (46.5%). Prefer `vector` or `temperature`.
