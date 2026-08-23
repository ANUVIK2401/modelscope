"""Multi-objective Pareto analysis with uncertainty awareness.

Two changes over the previous version:

1. Failed rows are dropped and environment hashes are checked. Merging results
   produced under different bitsandbytes builds and calling the union a frontier
   compares numbers that were never comparable.

2. A statistically-aware frontier. The naive frontier treats a 0.001 accuracy
   edge as domination, so noise decides which config is "optimal". The
   epsilon-dominance variant requires a config to beat another by more than the
   measurement uncertainty before domination is granted.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from analysis.paths import BENCHMARK_CSV, QUALITY_CSV

# A config must win by more than this to dominate on each axis. Accuracy epsilon
# reflects the MMLU standard error at the eval size used here.
DEFAULT_EPSILON: dict[str, float] = {
    "tokens_per_sec_batch1": 0.5,
    "memory_mb": 50.0,
    "mmlu_accuracy": 0.03,
}


def load_results(benchmark_csv=BENCHMARK_CSV, quality_csv=QUALITY_CSV) -> pd.DataFrame:
    """Merge benchmark and quality results, keeping only successful rows."""
    for path in (benchmark_csv, quality_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run benchmarks/runner.py and eval/harness.py first"
            )

    bench = pd.read_csv(benchmark_csv)
    qual = pd.read_csv(quality_csv)

    for name, df in (("benchmark", bench), ("quality", qual)):
        if "status" in df.columns:
            failed = df[df["status"] != "ok"]
            if not failed.empty:
                print(f"WARNING: dropping {len(failed)} failed {name} rows: "
                      f"{sorted(failed['variant'])}")

    bench = bench[bench.get("status", "ok") == "ok"] if "status" in bench else bench
    qual = qual[qual.get("status", "ok") == "ok"] if "status" in qual else qual

    merged = bench.merge(qual, on=["variant", "model", "config"],
                         how="inner", suffixes=("", "_qual"))
    if merged.empty:
        raise ValueError(
            "Merged results are empty - variant keys do not match across CSVs"
        )

    # Results from different software/hardware stacks are not comparable.
    for col in ("env_hash", "env_hash_qual"):
        if col in merged.columns and merged[col].nunique() > 1:
            print(f"WARNING: rows span multiple environments ({col}): "
                  f"{sorted(merged[col].unique())}. Cross-environment comparison "
                  "is not valid; re-run the affected variants on one stack.")

    # Canonical memory column: weights measured at load.
    if "weight_memory_mb" in merged.columns and "memory_mb" not in merged.columns:
        merged["memory_mb"] = merged["weight_memory_mb"]

    return merged


def _dominates(a: np.ndarray, b: np.ndarray, epsilon: np.ndarray) -> bool:
    """True if a dominates b by more than epsilon on at least one axis,
    and is not worse than b by more than epsilon on any axis."""
    not_worse = np.all(a >= b - epsilon)
    strictly_better = np.any(a > b + epsilon)
    return bool(not_worse and strictly_better)


def find_pareto_frontier(
    df: pd.DataFrame,
    objectives: Sequence[str] = ("tokens_per_sec_batch1", "memory_mb", "mmlu_accuracy"),
    epsilon: dict[str, float] | None = None,
    use_epsilon: bool = True,
) -> pd.DataFrame:
    """Mark non-dominated configurations.

    memory_mb is minimized (negated internally); the others are maximized.
    With use_epsilon=True, domination requires exceeding the measurement noise,
    so the frontier does not shuffle between runs on sub-noise differences.
    """
    missing = [c for c in objectives if c not in df.columns]
    if missing:
        raise KeyError(f"Missing objective columns: {missing}")

    eps_map = {**DEFAULT_EPSILON, **(epsilon or {})}

    columns = []
    eps_values = []
    for name in objectives:
        values = df[name].to_numpy(dtype=float)
        # Minimize memory: negate so every axis is "higher is better".
        columns.append(-values if name == "memory_mb" else values)
        eps_values.append(eps_map.get(name, 0.0) if use_epsilon else 0.0)

    points = np.column_stack(columns)
    eps = np.array(eps_values, dtype=float)

    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and _dominates(points[j], points[i], eps):
                is_pareto[i] = False
                break

    result = df.copy()
    result["is_pareto"] = is_pareto
    return result
