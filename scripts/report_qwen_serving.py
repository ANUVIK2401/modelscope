"""Create a report from one completed paired Qwen3 streaming job."""

import argparse
import json
import random
import statistics
from pathlib import Path

from benchmarks.qwen_serving import CONCURRENCIES, SLO_MS, percentile

QUALITY_FLOOR = -0.05


def read_conditions(folder):
    conditions = {}
    for file in folder.glob("*-c*-r*.json"):
        result = json.loads(file.read_text())
        summary = result["summary"]
        key = summary["workload"], summary["concurrency"]
        conditions.setdefault(key, []).append(result)
    return conditions


def score_quality(root):
    def load(variant):
        rows = json.loads((root / variant / "quality.json").read_text())
        return {row["id"]: row for row in rows}

    bf16, awq = load("bf16"), load("awq")
    if bf16.keys() != awq.keys() or not bf16:
        raise ValueError("Quality rows must contain identical nonempty case IDs.")
    differences = []
    for key in sorted(bf16):
        left, right = bf16[key], awq[key]
        if (left["prompt"], left["expected"]) != (right["prompt"], right["expected"]):
            raise ValueError(f"Quality case {key} differs between the variants.")
        differences.append(int(right["correct"]) - int(left["correct"]))
    rng = random.Random(42)
    bootstrap = [
        statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(4000)
    ]
    return {
        "count": len(differences),
        "bf16": statistics.mean(row["correct"] for row in bf16.values()),
        "awq": statistics.mean(row["correct"] for row in awq.values()),
        "delta": statistics.mean(differences),
        "ci": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
    }


def group_metrics(repeats):
    summaries = [repeat["summary"] for repeat in repeats]
    requests = [row for repeat in repeats for row in repeat["rows"]]
    successful = [row for row in requests if row["error"] is None]
    ttft = [row["ttft_ms"] for row in successful]
    prompt = [row["prompt_tokens"] for row in successful]
    elapsed = sum(row["elapsed_s"] for row in summaries)
    return {
        "n": len(requests),
        "errors": len(requests) - len(successful),
        "p50": percentile(ttft, 0.50),
        "p95": percentile(ttft, 0.95),
        "goodput": sum(row["ttft_ms"] <= summaries[0]["ttft_target_ms"]
                       for row in successful) / elapsed,
        "toks": sum(row["output_tokens"] for row in successful) / elapsed,
        "prompt_tokens": [min(prompt), max(prompt)] if prompt else None,
        "repeat_p95": [row["ttft_p95_ms"] for row in summaries],
        "repeat_goodput": [row["goodput_rps"] for row in summaries],
    }


def plot(root, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained", sharey=False)
    colors = {"bf16": "#264d9b", "awq": "#b25122"}
    for axis, workload in zip(axes, SLO_MS, strict=True):
        for variant, color in colors.items():
            points = [metrics[variant][workload, c] for c in CONCURRENCIES]
            axis.plot(CONCURRENCIES, [point["p95"] for point in points], "o-",
                      label=variant.upper(), color=color)
            for concurrency, point in zip(CONCURRENCIES, points, strict=True):
                repeats = [float(value) for value in point["repeat_p95"] if value is not None]
                if repeats:
                    axis.vlines(concurrency, min(repeats), max(repeats), alpha=0.65, color=color)
        axis.axhline(SLO_MS[workload], linestyle="--", color="#666", label="TTFT target")
        axis.set(xlabel="Concurrent requests", ylabel="p95 TTFT (ms)", title=f"{workload.title()} prompts",
                 xticks=CONCURRENCIES)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(root / "ttft-by-concurrency.png", dpi=170)


def report_prefix(root):
    env = json.loads((root / "environment.json").read_text())
    lines = [
        "# Qwen3-8B AWQ prefix caching", "",
        f"GPU: {env['gpu']}. vLLM {env['vllm']}. Concurrency 4.", "",
        "| Prefix caching | Workload | Pass | Requests | Errors | p50 TTFT (ms) | "
        "p95 TTFT (ms) | Goodput (req/s) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for enabled in (False, True):
        conditions = read_conditions(root / f"awq-prefix-{enabled}")
        for workload in ("shared_prefix", "unique_prefix"):
            for condition in sorted(conditions[workload, 4], key=lambda row: row["summary"]["repeat"]):
                result = condition["summary"]
                stage = "Repeat" if result["repeat"] else "Cold start"
                p50 = result["ttft_p50_ms"]
                p95 = result["ttft_p95_ms"]
                if p50 is None or p95 is None:
                    raise ValueError(f"No successful {workload} streams with prefix cache {enabled}")
                lines.append(
                    f"| {'On' if enabled else 'Off'} | {workload.replace('_', ' ').title()} | "
                    f"{stage} | {result['requests']} | {result['errors']} | {p50:.2f} | "
                    f"{p95:.2f} | {result['goodput_rps']:.2f} |"
                )
    lines.extend([
        "",
        "The cold-start pass begins with no cached entries for that workload. "
        "Shared prefixes can warm during the pass. The repeat uses the same requests, "
        "so the repeat over unique prefixes can also hit entries from the first pass.",
        "",
        "[Reproduction and measurement definitions](../../docs/benchmark-reproduction.md)",
        "",
    ])
    (root / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--prefix", action="store_true")
    args = parser.parse_args()
    root = args.result_dir
    if args.prefix:
        report_prefix(root)
        return
    env = json.loads((root / "environment.json").read_text())
    metrics = {
        variant: {key: group_metrics(repeats) for key, repeats in read_conditions(root / variant).items()}
        for variant in ("bf16", "awq")
    }
    required = {(workload, concurrency) for workload in SLO_MS for concurrency in CONCURRENCIES}
    if any(rows.keys() != required for rows in metrics.values()):
        raise SystemExit("The serving matrix is incomplete. Preserve the raw outputs as a partial run.")
    quality = score_quality(root)
    plot(root, metrics)
    lines = [
        "# Qwen3-8B BF16 and AWQ streaming", "",
        f"GPU: {env['gpu']}. vLLM {env['vllm']}, CUDA {env['cuda']}.",
        "",
        "![p95 first-token latency by concurrency](ttft-by-concurrency.png)", "",
        "Vertical ranges show the observed p95s across repeat blocks.",
        "",
        "| Variant | Workload | Concurrency | Requests | Errors | p50 TTFT (ms) | p95 TTFT (ms) | "
        "Output tok/s | Goodput (req/s) | Input tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant, rows in metrics.items():
        for key, data in sorted(rows.items()):
            prompt_span = "–".join(map(str, data["prompt_tokens"])) if data["prompt_tokens"] else "—"
            p50, p95 = ("—" if data[k] is None else f"{data[k]:.1f}" for k in ("p50", "p95"))
            lines.append(
                f"| {variant.upper()} | {key[0]} | {key[1]} | {data['n']} | {data['errors']} | "
                f"{p50} | {p95} | {data['toks']:.2f} | {data['goodput']:.2f} | {prompt_span} |"
            )
    lines.extend([
        "",
        f"Exact match on {quality['count']} paired prompts was {quality['bf16']:.1%} for BF16 "
        f"and {quality['awq']:.1%} for AWQ. The AWQ-minus-BF16 difference was "
        f"{quality['delta'] * 100:+.1f} points "
        f"(paired bootstrap 95% interval {quality['ci'][0] * 100:+.1f} to "
        f"{quality['ci'][1] * 100:+.1f} points). "
        "The set contains synthetic factual extraction and arithmetic tasks.",
        "",
    ])
    if quality["delta"] < QUALITY_FLOOR:
        lines.extend(["AWQ exceeded the five-point quality-loss threshold. BF16 remains the reference.", ""])
    else:
        for workload, target in SLO_MS.items():
            candidates = [
                (row["goodput"], variant, concurrency)
                for variant, rows in metrics.items()
                for (kind, concurrency), row in rows.items()
                if kind == workload and not row["errors"]
            ]
            if candidates and max(candidates)[0] > 0:
                goodput, variant, concurrency = max(candidates)
                lines.append(
                    f"- **{workload.title()} prompts ({target} ms target).** {variant.upper()} at "
                    f"concurrency {concurrency} had the highest observed goodput, {goodput:.2f} req/s."
                )
                observed_p95 = metrics[variant][workload, concurrency]["p95"]
                if observed_p95 > target:
                    compliant = [
                        (rate, name, c) for rate, name, c in candidates
                        if metrics[name][workload, c]["p95"] <= target
                    ]
                    message = f"  Its p95 was {observed_p95:.1f} ms."
                    if compliant:
                        compliant_rate, compliant_variant, compliant_c = max(compliant)
                        message += (
                            f" With a p95 ceiling of {target} ms, choose {compliant_variant.upper()} "
                            f"at concurrency {compliant_c} for {compliant_rate:.2f} req/s."
                        )
                    lines.append(message)
            else:
                lines.append(f"- **{workload.title()} prompts.** No error-free condition met the TTFT target.")
    lines.extend([
        "",
        "See [the benchmark protocol](../../docs/benchmark-reproduction.md) for the request and scoring "
        "definitions. Per-request samples and repeat blocks are in the variant folders.",
        "",
    ])
    (root / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
