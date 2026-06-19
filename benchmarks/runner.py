import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmarks.latency import measure_latency
from benchmarks.throughput import measure_throughput
from models.configs import BATCH_SIZES, BENCHMARK_PROMPTS, MODEL_REGISTRY
from models.loader import load_model, unload_model

_CSV_PATH = Path("results/benchmark_results.csv")
_CSV_COLUMNS = [
    "variant",
    "model",
    "config",
    "memory_mb",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tokens_per_sec_batch1",
    "tokens_per_sec_batch4",
    "tokens_per_sec_batch16",
    "tokens_per_sec_batch32",
]


def _append_row(row: dict[str, Any]) -> None:
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _CSV_PATH.exists()
    with _CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_all_benchmarks() -> None:
    prompt = BENCHMARK_PROMPTS[0]

    for variant_key in tqdm(MODEL_REGISTRY, desc="variants", unit="variant"):
        # "llama_fp16" → model_name="llama", config_name="fp16"
        # "llama_int4_nf4" → model_name="llama", config_name="int4_nf4"
        model_name, config_name = variant_key.split("_", 1)

        model, tokenizer, metadata = load_model(variant_key)

        latency = measure_latency(model, tokenizer, prompt)
        throughput = measure_throughput(model, tokenizer, BATCH_SIZES)

        unload_model(model)

        row: dict[str, Any] = {
            "variant": variant_key,
            "model": model_name,
            "config": config_name,
            "memory_mb": metadata["memory_delta_mb"],
            "ttft_p50_ms": latency["ttft_p50_ms"],
            "ttft_p95_ms": latency["ttft_p95_ms"],
            "tokens_per_sec_batch1": throughput.get(1),
            "tokens_per_sec_batch4": throughput.get(4),
            "tokens_per_sec_batch16": throughput.get(16),
            "tokens_per_sec_batch32": throughput.get(32),
        }
        _append_row(row)

        tqdm.write(
            f"  {variant_key}: memory={row['memory_mb']}MB "
            f"ttft_p50={row['ttft_p50_ms']}ms "
            f"tps_b1={row['tokens_per_sec_batch1']}"
        )
