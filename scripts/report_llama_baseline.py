"""Render the Colab T4 offline baseline from the two original result files."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "t4-llama-baseline"


def get_result(path, variant):
    with path.open(newline="") as source:
        return next(
            row for row in csv.DictReader(source)
            if row["variant"] == variant and row["status"] == "ok"
        )


def main():
    fp16 = get_result(ROOT / "initial.csv", "llama_fp16")
    nf4 = get_result(ROOT / "isolated-nf4.csv", "llama_nf4")
    if fp16["env_hash"] != nf4["env_hash"] or fp16["model_id"] != nf4["model_id"]:
        raise SystemExit("The baseline rows must share one model and environment.")
    env = json.loads((ROOT / "environment.json").read_text())
    revision = (ROOT / "model-revision.txt").read_text().strip()
    lines = [
        "# Llama 3.2 3B offline benchmark on Colab T4",
        "",
        f"Measured on one {env['gpu_name']} (compute capability {env['compute_capability']}) "
        f"in the same Colab runtime. Environment hash `{fp16['env_hash']}`. "
        f"The runtime cached Hugging Face revision `{revision}`.",
        "",
        "| Metric | FP16 | NF4 | NF4 change |",
        "|---|---:|---:|---:|",
    ]
    metrics = {
        "Post-load GPU driver VRAM (MiB)": "weight_memory_mb",
        "Post-load PyTorch allocated VRAM (MiB)": "weight_memory_mb_allocator",
        "Batch 1 throughput (output tok/s)": "tokens_per_sec_batch1",
        "Batch 4 throughput (output tok/s)": "tokens_per_sec_batch4",
        "Batch 16 throughput (output tok/s)": "tokens_per_sec_batch16",
        "Batch 32 throughput (output tok/s)": "tokens_per_sec_batch32",
        "One-token generation p50 (ms)": "ttft_p50_ms",
        "One-token generation p95 (ms)": "ttft_p95_ms",
    }
    for label, key in metrics.items():
        left, right = float(fp16[key]), float(nf4[key])
        lines.append(f"| {label} | {left:,.2f} | {right:,.2f} | {(right / left - 1):+.1%} |")
    lines.extend([
        "",
        "FP16 delivered higher throughput at every batch size measured. "
        "NF4 reduced post-load VRAM under both memory readings. "
        "This comparison includes one successful run per variant.",
        "",
        "The post-load driver figure is the change in device-wide used VRAM across model load. "
        "It includes allocator reservations and any other device activity during that window. "
        "The PyTorch figure is the change in live allocations tracked by its allocator. "
        "The latency rows time a separate one-token CUDA generation call after warm-up. "
        "HTTP request time, server queues, and streaming are outside that measurement. "
        "Throughput measures batches of identical prompts with a fixed generation length. "
        "No quality scores were collected for this comparison.",
        "",
        "The initial NF4 pass stopped at the memory sanity check. The isolated retry used "
        "the cached weights in a fresh Python process and passed the same check. "
        "The magnitude of the initial reading alone cannot establish its cause.",
        "",
        f"Stack: Python {env['python']}, PyTorch {env['torch']}, CUDA {env['cuda']}, "
        f"Transformers {env['transformers']}, bitsandbytes {env['bitsandbytes']}, "
        f"Accelerate {env['accelerate']}.",
        "",
        "[Initial CSV](initial.csv) · [Isolated NF4 CSV](isolated-nf4.csv) · "
        "[Retry log](isolated-nf4.log) · [FastAPI GPU smoke](fastapi-smoke.log) · "
        "[Environment](environment.json) · "
        "[Reproduction](../../docs/benchmark-reproduction.md)",
        "",
    ])
    (ROOT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
