"""Deployment recommendations, with uncertainty carried through.

Changes from the previous version: the phantom `relevance_score` column is gone
(nothing ever produced it -- the BERTScore module was never wired into the
harness, so this KeyError'd on real data), the MMLU quality floor is now
relative to the FP16 baseline instead of an absolute 0.65 that no variant could
meet, and every recommendation states its accuracy confidence interval.
"""

import json
from typing import Any

import pandas as pd

from analysis.paths import RECOMMENDATION_JSON

# A config qualifies as quality-preserving if its accuracy is within this
# absolute margin of its family's FP16 baseline. Relative to baseline rather
# than an absolute threshold: the question is what quantization costs, not
# whether the base model is good.
_MAX_ACCURACY_DROP = 0.05

_FIELDS = (
    "variant", "model", "config", "memory_mb", "tokens_per_sec_batch1",
    "mmlu_accuracy", "mmlu_ci_low", "mmlu_ci_high", "ttft_p50_ms",
    "itl_p50_ms", "consistency_score", "perplexity", "ece",
)


def _record(row: pd.Series, rationale: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in _FIELDS:
        if field in row.index and pd.notna(row[field]):
            value = row[field]
            out[field] = (
                round(float(value), 4) if isinstance(value, (int, float)) else str(value)
            )
    out["rationale"] = rationale
    return out


def _accuracy_interval(row: pd.Series) -> str:
    if "mmlu_ci_low" in row.index and pd.notna(row.get("mmlu_ci_low")):
        return f"MMLU {row['mmlu_accuracy']:.3f} [{row['mmlu_ci_low']:.3f}, {row['mmlu_ci_high']:.3f}]"
    return f"MMLU {row['mmlu_accuracy']:.3f}"


def _minmax(series: pd.Series) -> pd.Series:
    low, high = series.min(), series.max()
    if high == low:
        return pd.Series(0.5, index=series.index)
    return (series - low) / (high - low)


def generate_recommendation(df: pd.DataFrame,
                            output_path=RECOMMENDATION_JSON) -> dict[str, Any]:
    """Recommend a config for latency-first, memory-first, and balanced goals."""
    if df.empty:
        raise ValueError("Cannot recommend from an empty results frame")

    # Per-family FP16 baseline accuracy, for the relative quality floor.
    baselines = (
        df[df["config"] == "fp16"].set_index("model")["mmlu_accuracy"].to_dict()
    )
    df = df.copy()
    df["baseline_accuracy"] = df["model"].map(baselines)
    df["accuracy_drop"] = df["baseline_accuracy"] - df["mmlu_accuracy"]

    fastest = df.loc[df["tokens_per_sec_batch1"].idxmax()]
    latency = _record(fastest, (
        f"Highest throughput: {fastest['tokens_per_sec_batch1']:.1f} tok/s at batch 1, "
        f"{fastest['memory_mb']:.0f} MB weights, {_accuracy_interval(fastest)}."
    ))

    # The memory recommendation is about what quantization costs, so the floor
    # is applied to quantized configs. FP16 is the reference point and always
    # has zero drop; letting it self-qualify would make the floor unreachable
    # and mask the case where every quantized option is too lossy.
    quantized = df[df["config"] != "fp16"]
    quality_ok = quantized[quantized["accuracy_drop"] <= _MAX_ACCURACY_DROP]
    if quality_ok.empty:
        quality_ok = df
        note = (f"no quantized config stayed within {_MAX_ACCURACY_DROP} of its "
                "FP16 baseline; quality floor relaxed")
    else:
        note = f"within {_MAX_ACCURACY_DROP} accuracy of FP16 baseline"

    smallest = quality_ok.loc[quality_ok["memory_mb"].idxmin()]
    memory = _record(smallest, (
        f"Smallest quality-preserving footprint: {smallest['memory_mb']:.0f} MB weights "
        f"({note}), {smallest['tokens_per_sec_batch1']:.1f} tok/s, "
        f"{_accuracy_interval(smallest)}."
    ))

    composite = (
        _minmax(df["tokens_per_sec_batch1"])
        + _minmax(-df["memory_mb"])
        + _minmax(df["mmlu_accuracy"])
    )
    balanced_row = df.loc[composite.idxmax()]
    balanced = _record(balanced_row, (
        f"Best equal-weighted composite of speed, memory and accuracy "
        f"({composite.max():.3f}/3.000): {balanced_row['tokens_per_sec_batch1']:.1f} tok/s, "
        f"{balanced_row['memory_mb']:.0f} MB, {_accuracy_interval(balanced_row)}."
    ))

    recommendation = {
        "latency_priority": latency,
        "memory_priority": memory,
        "balanced": balanced,
        "quality_floor": {
            "rule": f"accuracy within {_MAX_ACCURACY_DROP:.2f} of same-family FP16",
            "baselines": {k: round(float(v), 4) for k, v in baselines.items()},
        },
        "caveat": (
            "Composite weighting is an arbitrary equal-weight choice, not an "
            "empirical result. Differences smaller than the reported confidence "
            "intervals should not drive a deployment decision."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(recommendation, indent=2))
    return recommendation
