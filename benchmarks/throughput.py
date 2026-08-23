"""Batched throughput and KV-cache memory.

Two corrections over the previous version:

1. Batch sizes [1,4,16,32] were declared in configs but never used -- every
   published throughput number was batch 1, the least deployment-relevant
   setting. Serving systems run batched; the memory ranking between configs can
   invert once the KV cache is counted, because 4-bit shrinks weights but leaves
   the FP16 cache untouched.

2. Left padding. Decoder-only models must left-pad for batched generation or
   they decode from pad tokens. The tokenizer is configured in the loader; this
   module asserts it, because a silent right-pad corrupts every batched number
   without raising.
"""

from typing import Any

import torch

from analysis.statistics import summarize_runs

_MAX_NEW_TOKENS = 100
_WARMUP_RUNS = 2
_MEASURED_RUNS = 3


def _device_used_mb() -> float:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return (total_bytes - free_bytes) / (1024 ** 2)


@torch.no_grad()
def measure_throughput(
    model: Any,
    tokenizer: Any,
    prompt: str,
    batch_sizes: list[int],
) -> dict[str, Any]:
    """Throughput and peak memory at each batch size.

    Returns per-batch throughput plus the incremental memory that batch level
    costs. An OOM at a given batch size is recorded as a result, not an error:
    "this config cannot serve batch 32 on a T4" is a finding.
    """
    if getattr(tokenizer, "padding_side", None) != "left":
        raise ValueError(
            "tokenizer.padding_side must be 'left' for batched decoder-only "
            "generation; right padding silently corrupts batched output."
        )

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    results: dict[str, Any] = {}

    for batch_size in batch_sizes:
        inputs = tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
        ).to(device)
        input_len = int(inputs["input_ids"].shape[1])

        try:
            for _ in range(_WARMUP_RUNS):
                model.generate(
                    **inputs, max_new_tokens=8, min_new_tokens=8,
                    do_sample=False, use_cache=True, pad_token_id=pad_id,
                )

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem_before = _device_used_mb()

            per_run_tps: list[float] = []
            for _ in range(_MEASURED_RUNS):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                out = model.generate(
                    **inputs,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    min_new_tokens=_MAX_NEW_TOKENS,
                    do_sample=False, use_cache=True, pad_token_id=pad_id,
                )
                end.record()
                torch.cuda.synchronize()
                elapsed_s = start.elapsed_time(end) / 1000.0
                n_new = (int(out.shape[1]) - input_len) * batch_size
                per_run_tps.append(n_new / elapsed_s)

            mem_peak = _device_used_mb()
            summary = summarize_runs(per_run_tps)

            results[f"tokens_per_sec_batch{batch_size}"] = summary["mean"]
            results[f"tokens_per_sec_batch{batch_size}_ci_low"] = summary["ci_low"]
            results[f"tokens_per_sec_batch{batch_size}_ci_high"] = summary["ci_high"]
            results[f"peak_memory_batch{batch_size}_mb"] = round(mem_peak, 1)
            results[f"kv_overhead_batch{batch_size}_mb"] = round(mem_peak - mem_before, 1)
            results[f"oom_batch{batch_size}"] = False

        except torch.cuda.OutOfMemoryError:
            # A config that cannot serve this batch size is a real deployment
            # constraint. Record it and continue rather than losing the variant.
            results[f"tokens_per_sec_batch{batch_size}"] = None
            results[f"peak_memory_batch{batch_size}_mb"] = None
            results[f"kv_overhead_batch{batch_size}_mb"] = None
            results[f"oom_batch{batch_size}"] = True
            torch.cuda.empty_cache()

        finally:
            torch.cuda.empty_cache()

    return results
