# Calibration report: Qwen/Qwen2.5-0.5B-Instruct (inline layout)

| set | labels | rotations | method | acc | ECE | Brier | NLL | s/req |
|---|---|---|---|---|---|---|---|---|
| ag_news | letters | 1 | raw | 0.800 | 0.159 | 0.313 | 0.773 | 0.35 |
| ag_news | letters | 1 | temperature | 0.800 | 0.086 | 0.265 | 0.498 | 0.35 |
| ag_news | letters | 1 | vector | 0.840 | 0.049 | 0.239 | 0.474 | 0.35 |
| ag_news | letters | 1 | contextual | 0.785 | 0.058 | 0.331 | 0.647 | 0.35 |
| ag_news | names | 1 | raw | 0.755 | 0.211 | 0.428 | 1.321 | 0.34 |
| ag_news | names | 1 | temperature | 0.755 | 0.104 | 0.321 | 0.620 | 0.34 |
| ag_news | names | 1 | vector | 0.855 | 0.077 | 0.243 | 0.469 | 0.34 |
| ag_news | names | 1 | contextual | 0.755 | 0.096 | 0.332 | 0.638 | 0.34 |
| ag_news | letters | 2 | raw | 0.795 | 0.146 | 0.345 | 0.996 | 0.72 |
| ag_news | letters | 2 | temperature | 0.795 | 0.082 | 0.296 | 0.565 | 0.72 |
| ag_news | letters | 2 | vector | 0.835 | 0.074 | 0.257 | 0.507 | 0.72 |
| ag_news | letters | 2 | contextual | 0.780 | 0.052 | 0.326 | 0.637 | 0.72 |
| ag_news | names | 2 | raw | 0.745 | 0.143 | 0.415 | 1.511 | 0.66 |
| ag_news | names | 2 | temperature | 0.745 | 0.096 | 0.336 | 0.631 | 0.66 |
| ag_news | names | 2 | vector | 0.815 | 0.055 | 0.255 | 0.491 | 0.66 |
| ag_news | names | 2 | contextual | 0.810 | 0.071 | 0.289 | 0.555 | 0.66 |
| sst2 | letters | 1 | raw | 0.885 | 0.120 | 0.182 | 0.311 | 0.17 |
| sst2 | letters | 1 | temperature | 0.885 | 0.077 | 0.170 | 0.278 | 0.17 |
| sst2 | letters | 1 | vector | 0.895 | 0.053 | 0.174 | 0.285 | 0.17 |
| sst2 | letters | 1 | contextual | 0.465 | 0.292 | 0.479 | 0.661 | 0.17 |
| sst2 | letters | 2 | raw | 0.890 | 0.069 | 0.162 | 0.290 | 0.33 |
| sst2 | letters | 2 | temperature | 0.890 | 0.048 | 0.154 | 0.257 | 0.33 |
| sst2 | letters | 2 | vector | 0.895 | 0.048 | 0.165 | 0.269 | 0.33 |
| sst2 | letters | 2 | contextual | 0.465 | 0.311 | 0.472 | 0.652 | 0.33 |
| boolq | letters | 1 | raw | 0.655 | 0.167 | 0.504 | 0.751 | 0.55 |
| boolq | letters | 1 | temperature | 0.655 | 0.054 | 0.449 | 0.640 | 0.55 |
| boolq | letters | 1 | vector | 0.670 | 0.059 | 0.400 | 0.586 | 0.55 |
| boolq | letters | 1 | contextual | 0.660 | 0.103 | 0.421 | 0.610 | 0.55 |
| boolq | letters | 2 | raw | 0.675 | 0.160 | 0.452 | 0.717 | 1.21 |
| boolq | letters | 2 | temperature | 0.675 | 0.058 | 0.418 | 0.605 | 1.21 |
| boolq | letters | 2 | vector | 0.700 | 0.073 | 0.399 | 0.585 | 1.21 |
| boolq | letters | 2 | contextual | 0.660 | 0.100 | 0.421 | 0.610 | 1.21 |
