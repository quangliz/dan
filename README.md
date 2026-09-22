# dan

Fast, easy System One model inference and serving — like vLLM, but for decisions.

dan turns any open-weight causal LM into a System One model: typed, calibrated answers
(`noul` / `choice` / `score`, Jev-compatible) read from label-token probabilities in a
single prefill — no tokens generated.
