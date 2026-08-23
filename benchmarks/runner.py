"""Benchmark orchestrator: one implementation, imported everywhere.

Replaces the previous Colab script that was written to disk as a string literal
from a notebook cell. That path was untested, used hardcoded /content paths, and
produced every published number -- including the mislabelled TTFT column and the
24 MB reading for a 2B model. The notebook now imports this module.

Failure policy: a variant that raises records a row with `status="failed"` and
the error text, rather than being silently skipped. The previous
`except Exception: continue` left no trace, so the resume logic re-ran failures
forever and the Pareto analysis silently ran over a subset.
"""

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from analysis.paths import BENCHMARK_CSV, ENV_JSON, ensure_results_dir
from benchmarks.latency import measure_latency
from benchmarks.throughput import measure_throughput
from models.configs import (
    ALL_VARIANTS,
    BATCH_SIZES,
    BENCHMARK_PROMPTS,
    MODEL_REGISTRY,
)
from models.env import capture_environment, environment_hash, set_global_seed
from models.loader import measure_kv_cache_mb, measure_load_memory, unload_model


def completed_variants(csv_path: Path) -> set[str]:
    """Variants already recorded with status=ok. Failed rows are retried."""
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {
            row["variant"] for row in csv.DictReader(f)
            if row.get("status") == "ok"
        }


def append_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one row, writing the header on first use.

    Written immediately after each variant so a Colab disconnect at variant 7
    of 8 costs one variant, not the whole run.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def benchmark_variant(variant_key: str, env_hash: str) -> dict[str, Any]:
    """Full benchmark for one variant. Raises on unrecoverable failure."""
    _model_id, family, config_name, _bnb = MODEL_REGISTRY[variant_key]

    model, tokenizer, meta = measure_load_memory(variant_key)
    try:
        latency = measure_latency(model, tokenizer, BENCHMARK_PROMPTS, runs=5)
        throughput = measure_throughput(
            model, tokenizer, BENCHMARK_PROMPTS[0], BATCH_SIZES
        )
        # KV cache at a realistic serving shape, reported separately from
        # weights so the deployment-footprint claim is decomposable.
        kv_cache_mb = measure_kv_cache_mb(model, tokenizer, batch_size=8, seq_len=512)
    finally:
        unload_model(model)

    row: dict[str, Any] = {
        "variant": variant_key,
        "model": family,
        "config": config_name,
        "status": "ok",
        "error": "",
        "env_hash": env_hash,
        **{k: v for k, v in meta.items() if k not in ("variant", "model", "config")},
        **latency,
        **throughput,
        "kv_cache_b8_s512_mb": kv_cache_mb,
    }
    return row


def run(variants: list[str], csv_path: Path = BENCHMARK_CSV,
        resume: bool = True) -> None:
    set_global_seed()
    ensure_results_dir()

    env = capture_environment()
    env_hash = environment_hash(env)
    ENV_JSON.write_text(json.dumps(env, indent=2))
    print(f"Environment {env_hash}: {env['gpu_name']}, torch {env['torch']}, "
          f"bitsandbytes {env['bitsandbytes']}")
    if not env["is_ampere_or_newer"]:
        print(
            "NOTE: pre-Ampere GPU. INT8 throughput here reflects bitsandbytes' "
            "LLM.int8() mixed-precision path on hardware without wide INT8 "
            "tensor-core support; these results do not generalize to A100/L4."
        )

    done = completed_variants(csv_path) if resume else set()
    if done:
        print(f"Resuming, already complete: {sorted(done)}")

    for variant_key in tqdm(variants, desc="variants"):
        if variant_key in done:
            continue
        try:
            row = benchmark_variant(variant_key, env_hash)
            append_row(csv_path, row)
            print(
                f"{variant_key}: {row['weight_memory_mb']} MB weights, "
                f"{row['tokens_per_sec_batch1']} tok/s, "
                f"TTFT p50 {row['ttft_p50_ms']} ms"
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            traceback.print_exc()
            _model_id, family, config_name, _bnb = MODEL_REGISTRY[variant_key]
            append_row(csv_path, {
                "variant": variant_key,
                "model": family,
                "config": config_name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "env_hash": env_hash,
            })
            print(f"FAILED {variant_key}: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference benchmarks.")
    parser.add_argument("--variants", nargs="*", default=ALL_VARIANTS,
                        help="Variant keys to benchmark (default: all).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run variants already marked complete.")
    parser.add_argument("--output", type=Path, default=BENCHMARK_CSV)
    args = parser.parse_args()
    run(args.variants, csv_path=args.output, resume=not args.no_resume)


if __name__ == "__main__":
    main()
