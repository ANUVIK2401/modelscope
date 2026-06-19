from pathlib import Path

import numpy as np
import pandas as pd

_BENCHMARK_CSV = Path("results/benchmark_results.csv")
_QUALITY_CSV = Path("results/quality_results.csv")


def load_results() -> pd.DataFrame:
    for path in (_BENCHMARK_CSV, _QUALITY_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run benchmarks/runner.py and eval/harness.py first"
            )

    bench = pd.read_csv(_BENCHMARK_CSV)
    qual = pd.read_csv(_QUALITY_CSV)

    # Both CSVs carry 'model' and 'config'; merge on all three shared keys
    # so the join is unambiguous and the columns aren't duplicated.
    merged = bench.merge(qual, on=["variant", "model", "config"], how="inner")
    if merged.empty:
        raise ValueError("Merged DataFrame is empty — check that variant keys match across CSVs")
    return merged


def find_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Non-dominated set across three objectives:
      maximize tokens_per_sec_batch1
      minimize memory_mb       (negated to maximize)
      maximize mmlu_accuracy

    Point i is Pareto-optimal if no other point j weakly dominates it on all
    three objectives and is strictly better on at least one.
    """
    objectives = np.column_stack([
        df["tokens_per_sec_batch1"].to_numpy(dtype=float),
        -df["memory_mb"].to_numpy(dtype=float),
        df["mmlu_accuracy"].to_numpy(dtype=float),
    ])

    n = len(objectives)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            weakly_dominates = np.all(objectives[j] >= objectives[i])
            strictly_better = np.any(objectives[j] > objectives[i])
            if weakly_dominates and strictly_better:
                is_pareto[i] = False
                break

    result = df.copy()
    result["is_pareto"] = is_pareto
    return result
