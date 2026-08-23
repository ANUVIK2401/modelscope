"""Predict quantization damage from weight statistics alone.

Tests the claim that layer sensitivity is forecastable without running an eval.
Validation is grouped by model family: training on Llama layers and testing on
Gemma layers answers the question that matters -- does this transfer to a model
you have not benchmarked? Random splits would leak, because layers within one
model are highly correlated with each other.

Baseline comparison is mandatory. A model that beats nothing has not shown that
its features carry signal, so every fit is reported against a mean predictor.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cheap features, all computable from an FP16 checkpoint with no inference.
FEATURES: list[str] = [
    "kurtosis",
    "skewness",
    "outlier_ratio",
    "block_dynamic_range",
    "block_range_p95",
    "std",
    "q99_abs",
    "n_params",
    "depth",
]

TARGET = "relative_frobenius_error"


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    # Kurtosis and outlier ratio span orders of magnitude; log1p keeps a few
    # extreme layers from dominating the fit.
    for column in ("kurtosis", "outlier_ratio", "n_params", "block_range_p95"):
        if column in frame.columns:
            frame[column] = np.log1p(frame[column].clip(lower=0))
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])


def fit_sensitivity_predictor(
    df: pd.DataFrame,
    method: str = "fp4",
    group_column: str = "model_family",
    seed: int = 42,
) -> dict[str, Any]:
    """Fit and validate the predictor with grouped cross-validation.

    Returns held-out R^2 and MAE against a mean-predictor baseline, plus
    permutation feature importances.
    """
    subset = _prepare(df[df["method"] == method])
    if subset.empty:
        raise ValueError(f"No rows for method '{method}'")

    X = subset[FEATURES].to_numpy(dtype=float)
    y = subset[TARGET].to_numpy(dtype=float)

    if group_column in subset.columns and subset[group_column].nunique() > 1:
        groups = subset[group_column].to_numpy()
        n_splits = int(subset[group_column].nunique())
        split_kind = f"leave-one-{group_column}-out"
    else:
        # Fall back to depth-based groups so layers from the same block stay
        # together; still avoids the worst leakage of a random split.
        groups = subset["depth"].to_numpy()
        n_splits = min(5, len(np.unique(groups)))
        split_kind = "grouped-by-depth"

    if n_splits < 2:
        raise ValueError("Need at least 2 groups for grouped cross-validation")

    models = {
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=seed, n_jobs=-1
        ),
    }

    splitter = GroupKFold(n_splits=n_splits)
    results: dict[str, Any] = {
        "method": method,
        "n_layers": int(len(subset)),
        "split": split_kind,
        "n_splits": n_splits,
        "models": {},
    }

    for name, estimator in models.items():
        y_true_all: list[float] = []
        y_pred_all: list[float] = []
        baseline_all: list[float] = []

        for train_idx, test_idx in splitter.split(X, y, groups):
            estimator.fit(X[train_idx], y[train_idx])
            predictions = estimator.predict(X[test_idx])
            y_true_all.extend(y[test_idx])
            y_pred_all.extend(predictions)
            # Baseline: predict the training mean for every held-out layer.
            baseline_all.extend([y[train_idx].mean()] * len(test_idx))

        results["models"][name] = {
            "r2_heldout": round(float(r2_score(y_true_all, y_pred_all)), 4),
            "mae_heldout": round(float(mean_absolute_error(y_true_all, y_pred_all)), 6),
            "r2_baseline": round(float(r2_score(y_true_all, baseline_all)), 4),
            "mae_baseline": round(float(mean_absolute_error(y_true_all, baseline_all)), 6),
            "beats_baseline": bool(
                mean_absolute_error(y_true_all, y_pred_all)
                < mean_absolute_error(y_true_all, baseline_all)
            ),
        }

    results["feature_importance"] = _permutation_importance(X, y, groups, splitter, seed)
    results["correlations"] = {
        feature: round(float(np.corrcoef(subset[feature], subset[TARGET])[0, 1]), 4)
        for feature in FEATURES
        if subset[feature].std() > 0
    }
    return results


def _permutation_importance(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                            splitter: Any, seed: int) -> dict[str, float]:
    """Held-out permutation importance: MAE increase when a feature is shuffled."""
    rng = np.random.default_rng(seed)
    model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                  random_state=seed, n_jobs=-1)

    base_scores: list[float] = []
    permuted_scores: dict[str, list[float]] = {f: [] for f in FEATURES}

    for train_idx, test_idx in splitter.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        X_test, y_test = X[test_idx], y[test_idx]
        base_mae = mean_absolute_error(y_test, model.predict(X_test))
        base_scores.append(base_mae)

        for i, feature in enumerate(FEATURES):
            shuffled = X_test.copy()
            shuffled[:, i] = rng.permutation(shuffled[:, i])
            permuted_scores[feature].append(
                mean_absolute_error(y_test, model.predict(shuffled))
            )

    base = float(np.mean(base_scores))
    return {
        feature: round(float(np.mean(scores) - base), 6)
        for feature, scores in sorted(
            permuted_scores.items(),
            key=lambda kv: -float(np.mean(kv[1])),
        )
    }


def compare_formats(df: pd.DataFrame) -> dict[str, Any]:
    """Does NF4 beat FP4, and does the gap depend on distribution shape?

    The QLoRA claim is that NF4's normal-quantile levels suit Gaussian weights.
    The corollary this tests: the advantage should grow with heavier tails,
    where uniform FP4 levels waste resolution on an empty range.
    """
    pivot = df.pivot_table(
        index=["layer", "model_family"] if "model_family" in df.columns else ["layer"],
        columns="method",
        values=TARGET,
    ).dropna()

    if "fp4" not in pivot.columns or "nf4" not in pivot.columns:
        raise ValueError("Both fp4 and nf4 errors required")

    advantage = (pivot["fp4"] - pivot["nf4"]) / pivot["fp4"]

    stats_source = df[df["method"] == "fp4"].set_index(
        ["layer", "model_family"] if "model_family" in df.columns else ["layer"]
    )
    aligned = stats_source.reindex(pivot.index)

    correlations = {}
    for feature in ("kurtosis", "block_dynamic_range", "outlier_ratio"):
        if feature in aligned.columns:
            values = np.log1p(aligned[feature].clip(lower=0))
            mask = values.notna() & advantage.notna()
            if mask.sum() > 2 and values[mask].std() > 0:
                correlations[feature] = round(
                    float(np.corrcoef(values[mask], advantage[mask])[0, 1]), 4
                )

    return {
        "n_layers": int(len(pivot)),
        "mean_fp4_error": round(float(pivot["fp4"].mean()), 6),
        "mean_nf4_error": round(float(pivot["nf4"].mean()), 6),
        "nf4_relative_advantage": round(float(advantage.mean()), 4),
        "nf4_wins_fraction": round(float((advantage > 0).mean()), 4),
        "advantage_vs_tail_correlation": correlations,
    }
