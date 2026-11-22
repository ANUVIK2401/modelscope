# Failed Qwen3-8B startup attempt (L4)

An earlier Cloud Run L4 attempt in which vLLM did not finish starting either variant
before the harness deadline (`failure.json` in each variant folder). No requests were
served, so there are no latency or quality numbers here. Kept so the successful runs in
`../l4-qwen-paired/` and `../l4-qwen-prefix/` are not read as the only attempts.
Server logs have ANSI colour codes removed; nothing else was altered.
