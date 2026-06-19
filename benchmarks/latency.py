import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MAX_NEW_TOKENS = 100


def measure_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    runs: int = 5,
) -> dict[str, float]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # TTFT: time from generate() call until exactly 1 new token is decoded.
    # max_new_tokens=1 captures prefill + one decode step with no loop overhead.
    ttft_samples_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        ttft_samples_ms.append((time.perf_counter() - t0) * 1000)

    # Full generation: used for ITL and tokens/sec.
    full_times_s: list[float] = []
    token_counts: list[int] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        full_times_s.append(time.perf_counter() - t0)
        token_counts.append(int(output_ids.shape[1]) - input_len)

    ttft_arr = np.array(ttft_samples_ms)
    ttft_p50_ms = float(np.percentile(ttft_arr, 50))
    ttft_p95_ms = float(np.percentile(ttft_arr, 95))

    # Per-run ITL = (full_time - mean_ttft) / (n_tokens - 1).
    # Using mean_ttft as the TTFT offset keeps variance in the ITL estimate
    # honest without requiring per-run prefill timing.
    mean_ttft_s = ttft_p50_ms / 1000
    itl_samples_ms = [
        ((ft - mean_ttft_s) / (nt - 1)) * 1000
        for ft, nt in zip(full_times_s, token_counts)
        if nt > 1
    ]
    itl_p50_ms = float(np.percentile(itl_samples_ms, 50)) if itl_samples_ms else 0.0

    mean_tokens = float(np.mean(token_counts))
    mean_full_s = float(np.mean(full_times_s))
    tokens_per_sec = mean_tokens / mean_full_s

    return {
        "ttft_p50_ms": round(ttft_p50_ms, 3),
        "ttft_p95_ms": round(ttft_p95_ms, 3),
        "itl_p50_ms": round(itl_p50_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }
