import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.configs import BENCHMARK_PROMPTS

_MAX_NEW_TOKENS = 100
_WARMUP_RUNS = 3
_MEASURED_RUNS = 5


def measure_throughput(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_sizes: list[int],
) -> dict[int, float]:
    device = next(model.parameters()).device
    prompt = BENCHMARK_PROMPTS[0]
    results: dict[int, float] = {}

    for batch_size in batch_sizes:
        inputs = tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        for _ in range(_WARMUP_RUNS):
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        total_tokens = 0
        total_time_s = 0.0
        for _ in range(_MEASURED_RUNS):
            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            total_time_s += time.perf_counter() - t0
            total_tokens += (output_ids.shape[1] - input_len) * batch_size

        results[batch_size] = round(total_tokens / total_time_s, 2)

    return results
