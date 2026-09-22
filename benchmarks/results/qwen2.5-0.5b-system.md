# Calibration report: Qwen/Qwen2.5-0.5B-Instruct (system layout)

| set | labels | rotations | method | acc | ECE | Brier | NLL | s/req |
|---|---|---|---|---|---|---|---|---|
| ag_news | letters | 1 | raw | 0.275 | 0.157 | 0.788 | 1.447 | 0.33 |
| ag_news | letters | 1 | temperature | 0.275 | 0.025 | 0.747 | 1.378 | 0.33 |
| ag_news | letters | 1 | vector | 0.340 | 0.042 | 0.742 | 1.371 | 0.33 |
| ag_news | letters | 1 | contextual | 0.220 | 0.035 | 0.750 | 1.386 | 0.33 |
| ag_news | names | 1 | raw | 0.320 | 0.212 | 0.789 | 1.452 | 0.29 |
| ag_news | names | 1 | temperature | 0.320 | 0.061 | 0.716 | 1.302 | 0.29 |
| ag_news | names | 1 | vector | 0.535 | 0.098 | 0.609 | 1.112 | 0.29 |
| ag_news | names | 1 | contextual | 0.240 | 0.061 | 0.740 | 1.366 | 0.29 |
| ag_news | letters | 2 | raw | 0.350 | 0.057 | 0.763 | 1.603 | 1.20 |
| ag_news | letters | 2 | temperature | 0.350 | 0.066 | 0.746 | 1.380 | 1.20 |
| ag_news | letters | 2 | vector | 0.370 | 0.051 | 0.738 | 1.365 | 1.20 |
| ag_news | letters | 2 | contextual | 0.330 | 0.039 | 0.746 | 1.380 | 1.20 |
| ag_news | letters | 4 | raw | 0.260 | 0.081 | 0.788 | 1.720 | 2.10 |
| ag_news | letters | 4 | temperature | 0.260 | 0.027 | 0.750 | 1.385 | 2.10 |
| ag_news | letters | 4 | vector | 0.290 | 0.051 | 0.747 | 1.381 | 2.10 |
| ag_news | letters | 4 | contextual | 0.260 | 0.031 | 0.750 | 1.385 | 2.10 |
| sst2 | letters | 1 | raw | 0.795 | 0.099 | 0.294 | 0.462 | 0.53 |
| sst2 | letters | 1 | temperature | 0.795 | 0.109 | 0.300 | 0.469 | 0.53 |
| sst2 | letters | 1 | vector | 0.790 | 0.090 | 0.309 | 0.480 | 0.53 |
| sst2 | letters | 1 | contextual | 0.460 | 0.210 | 0.499 | 0.690 | 0.53 |
| sst2 | letters | 2 | raw | 0.775 | 0.125 | 0.330 | 0.550 | 0.83 |
| sst2 | letters | 2 | temperature | 0.775 | 0.057 | 0.313 | 0.480 | 0.83 |
| sst2 | letters | 2 | vector | 0.815 | 0.090 | 0.279 | 0.441 | 0.83 |
| sst2 | letters | 2 | contextual | 0.445 | 0.328 | 0.511 | 0.704 | 0.83 |
| boolq | letters | 1 | raw | 0.665 | 0.097 | 0.441 | 0.652 | 1.28 |
| boolq | letters | 1 | temperature | 0.665 | 0.029 | 0.418 | 0.604 | 1.28 |
| boolq | letters | 1 | vector | 0.680 | 0.057 | 0.413 | 0.600 | 1.28 |
| boolq | letters | 1 | contextual | 0.685 | 0.029 | 0.417 | 0.605 | 1.28 |
| boolq | letters | 2 | raw | 0.615 | 0.109 | 0.459 | 0.684 | 1.87 |
| boolq | letters | 2 | temperature | 0.615 | 0.052 | 0.434 | 0.622 | 1.87 |
| boolq | letters | 2 | vector | 0.675 | 0.049 | 0.419 | 0.606 | 1.87 |
| boolq | letters | 2 | contextual | 0.660 | 0.033 | 0.425 | 0.614 | 1.87 |
