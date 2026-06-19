import json
from pathlib import Path
from typing import Any

import pandas as pd

_RECOMMENDATION_JSON = Path("results/recommendation.json")
_MMLU_MEMORY_THRESHOLD = 0.65


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _row_to_record(row: pd.Series, rationale: str) -> dict[str, Any]:
    return {
        "variant":              str(row["variant"]),
        "model":                str(row["model"]),
        "config":               str(row["config"]),
        "memory_mb":            round(float(row["memory_mb"]), 1),
        "tokens_per_sec_batch1": round(float(row["tokens_per_sec_batch1"]), 2),
        "mmlu_accuracy":        round(float(row["mmlu_accuracy"]), 4),
        "ttft_p50_ms":          round(float(row["ttft_p50_ms"]), 3),
        "consistency_score":    round(float(row["consistency_score"]), 4),
        "relevance_score":      round(float(row["relevance_score"]), 4),
        "rationale":            rationale,
    }


def generate_recommendation(df: pd.DataFrame) -> dict[str, Any]:
    # ── latency_priority: highest throughput regardless of cost ──────────────
    lat_row = df.loc[df["tokens_per_sec_batch1"].idxmax()]
    lat_rec = _row_to_record(
        lat_row,
        rationale=(
            f"Fastest inference: {lat_row['tokens_per_sec_batch1']:.0f} tok/s at batch 1, "
            f"{lat_row['memory_mb']:.0f} MB VRAM, TTFT p50 {lat_row['ttft_p50_ms']:.1f} ms."
        ),
    )

    # ── memory_priority: lowest VRAM while preserving quality ────────────────
    qualified = df[df["mmlu_accuracy"] > _MMLU_MEMORY_THRESHOLD]
    if qualified.empty:
        # Relax quality floor: just pick the lowest-memory variant overall.
        qualified = df
        quality_note = "(no variant exceeded the 0.65 MMLU threshold; floor relaxed)"
    else:
        quality_note = f"MMLU {qualified.loc[qualified['memory_mb'].idxmin(), 'mmlu_accuracy']:.3f} > {_MMLU_MEMORY_THRESHOLD}"

    mem_row = qualified.loc[qualified["memory_mb"].idxmin()]
    mem_rec = _row_to_record(
        mem_row,
        rationale=(
            f"Most memory-efficient: {mem_row['memory_mb']:.0f} MB VRAM, "
            f"{quality_note}, {mem_row['tokens_per_sec_batch1']:.0f} tok/s."
        ),
    )

    # ── balanced: highest composite of normalized speed + memory + quality ───
    norm_tps = _minmax(df["tokens_per_sec_batch1"])
    norm_mem = _minmax(-df["memory_mb"])       # negate: lower memory → higher score
    norm_mmlu = _minmax(df["mmlu_accuracy"])
    composite = norm_tps + norm_mem + norm_mmlu

    bal_row = df.loc[composite.idxmax()]
    bal_rec = _row_to_record(
        bal_row,
        rationale=(
            f"Best composite score: {bal_row['tokens_per_sec_batch1']:.0f} tok/s, "
            f"{bal_row['memory_mb']:.0f} MB VRAM, MMLU {bal_row['mmlu_accuracy']:.3f} — "
            f"normalized sum {composite.max():.3f}/3.000."
        ),
    )

    recommendation: dict[str, Any] = {
        "latency_priority": lat_rec,
        "memory_priority":  mem_rec,
        "balanced":         bal_rec,
    }

    _RECOMMENDATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    with _RECOMMENDATION_JSON.open("w") as f:
        json.dump(recommendation, f, indent=2)

    return recommendation
