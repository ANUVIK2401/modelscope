"""Latency measurement: TTFT and inter-token latency, measured separately.

The previous runner computed `(total_time / n_tokens) * 1000` and labelled it
`ttft_p50_ms`. That is mean inter-token latency, not time-to-first-token, and it
is algebraically just 1000/tokens_per_sec -- the reported latency column carried
no information the throughput column did not already have.

TTFT and ITL are genuinely different quantities driven by different hardware
behaviour: TTFT is prefill, compute-bound and parallel over the prompt; ITL is
decode, memory-bandwidth-bound and sequential. Quantization affects them in
opposite directions, which is precisely what makes measuring them separately
worthwhile.
"""

from typing import Any

import torch

from analysis.statistics import summarize_runs

_MAX_NEW_TOKENS = 100
_WARMUP_RUNS = 3


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _timed_generate(model: Any, inputs: dict, max_new_tokens: int,
                    pad_token_id: int) -> tuple[float, int]:
    """One generate() call, wall-clock timed with CUDA sync on both sides."""
    _sync()
    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if t0 is not None:
        t0.record()
    else:
        import time
        start = time.perf_counter()

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,  # defeat EOS so token count is fixed
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,
    )

    if t1 is not None:
        t1.record()
        torch.cuda.synchronize()
        elapsed_s = t0.elapsed_time(t1) / 1000.0
    else:
        import time
        elapsed_s = time.perf_counter() - start

    n_new = int(output_ids.shape[1]) - int(inputs["input_ids"].shape[1])
    return elapsed_s, n_new


def measure_latency(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    runs: int = 5,
) -> dict[str, Any]:
    """Measure TTFT and ITL across prompts.

    TTFT is the prefill-plus-one-decode-step time (`max_new_tokens=1`).
    ITL is derived per run as (full_time - ttft) / (n_tokens - 1), using that
    run's own TTFT rather than a pooled median, so the spread reported is real.
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    ttft_ms_all: list[float] = []
    itl_ms_all: list[float] = []
    tps_all: list[float] = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        for _ in range(_WARMUP_RUNS):
            _timed_generate(model, inputs, 8, pad_id)

        for _ in range(runs):
            ttft_s, _ = _timed_generate(model, inputs, 1, pad_id)
            full_s, n_tokens = _timed_generate(model, inputs, _MAX_NEW_TOKENS, pad_id)

            ttft_ms_all.append(ttft_s * 1000)
            if n_tokens > 1:
                itl_ms_all.append(((full_s - ttft_s) / (n_tokens - 1)) * 1000)
                tps_all.append(n_tokens / full_s)

    ttft = summarize_runs(ttft_ms_all)
    itl = summarize_runs(itl_ms_all)
    tps = summarize_runs(tps_all)

    return {
        "ttft_p50_ms": ttft["p50"],
        "ttft_p95_ms": ttft["p95"],
        "ttft_ci_low_ms": ttft["ci_low"],
        "ttft_ci_high_ms": ttft["ci_high"],
        "itl_p50_ms": itl["p50"],
        "itl_p95_ms": itl["p95"],
        "tokens_per_sec_batch1": tps["mean"],
        "tokens_per_sec_ci_low": tps["ci_low"],
        "tokens_per_sec_ci_high": tps["ci_high"],
        "tokens_per_sec_std": tps["std"],
        "n_latency_samples": ttft["n"],
    }
