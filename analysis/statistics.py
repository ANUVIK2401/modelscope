"""Uncertainty quantification for benchmark and eval results.

Every headline claim in this project must survive a confidence interval. On 100
MMLU questions the standard error is ~4.8 percentage points, so a "0.37 vs 0.35"
gap is indistinguishable from noise -- but reads as a finding if reported as
bare point estimates. These helpers make the noise floor explicit.
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats

DEFAULT_BOOTSTRAP = 10_000
DEFAULT_ALPHA = 0.05


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = DEFAULT_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns (mean, lo, hi)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    if arr.size == 1:
        v = float(arr[0])
        return v, v, v

    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def accuracy_ci(
    correct_flags: Sequence[int],
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation: it stays inside [0,1] and remains
    correct at the small n and extreme proportions this eval produces.
    """
    arr = np.asarray(correct_flags, dtype=float)
    n = arr.size
    if n == 0:
        return 0.0, 0.0, 0.0

    p = float(arr.mean())
    z = float(stats.norm.ppf(1 - alpha / 2))
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def paired_bootstrap_difference(
    flags_a: Sequence[int],
    flags_b: Sequence[int],
    n_resamples: int = DEFAULT_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> dict[str, float | bool]:
    """Paired bootstrap on accuracy difference (a - b), same question set.

    Pairing is what buys the power here: both variants answer identical
    questions, so per-question difficulty cancels and a much smaller true gap
    becomes detectable than an unpaired comparison would allow.
    """
    a = np.asarray(flags_a, dtype=float)
    b = np.asarray(flags_b, dtype=float)
    if a.size != b.size:
        raise ValueError(
            f"Paired test requires equal-length vectors, got {a.size} and {b.size}. "
            "Both variants must be scored on the same question set."
        )
    if a.size == 0:
        return {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "p_value": 1.0, "significant": False}

    diff = a - b
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    resampled = diff[idx].mean(axis=1)

    lo = float(np.percentile(resampled, 100 * alpha / 2))
    hi = float(np.percentile(resampled, 100 * (1 - alpha / 2)))

    # Two-sided p: fraction of centred resamples at least as extreme as observed.
    centred = resampled - observed
    p_value = float((np.abs(centred) >= abs(observed)).mean())

    return {
        "difference": round(observed, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_value": round(p_value, 4),
        # Significant when the CI excludes zero.
        "significant": bool(lo > 0 or hi < 0),
    }


def mcnemar_test(flags_a: Sequence[int], flags_b: Sequence[int]) -> dict[str, float | bool]:
    """McNemar's exact test for paired binary outcomes.

    The textbook test for "did these two classifiers differ on the same items".
    Only the discordant pairs carry information.
    """
    a = np.asarray(flags_a, dtype=int)
    b = np.asarray(flags_b, dtype=int)
    if a.size != b.size:
        raise ValueError("McNemar requires paired vectors of equal length")

    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    n_discordant = a_only + b_only

    if n_discordant == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0, "significant": False}

    # Exact binomial under H0: discordant pairs split 50/50.
    p_value = float(stats.binomtest(a_only, n_discordant, 0.5).pvalue)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "p_value": round(p_value, 4),
        "significant": bool(p_value < DEFAULT_ALPHA),
    }


def summarize_runs(values: Sequence[float]) -> dict[str, float]:
    """Point estimate plus spread for repeated timing measurements."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "p50": 0.0, "p95": 0.0, "n": 0}
    mean, lo, hi = bootstrap_ci(arr)
    return {
        "mean": round(mean, 3),
        "std": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 3),
        "ci_low": round(lo, 3),
        "ci_high": round(hi, 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "n": int(arr.size),
    }


def minimum_detectable_effect(n: int, p: float = 0.5, alpha: float = DEFAULT_ALPHA,
                              power: float = 0.8) -> float:
    """Smallest accuracy gap detectable at given n, alpha, power.

    Reported in the limitations section: it states, in advance, which
    comparisons this study is not powered to make.
    """
    if n <= 0:
        return 1.0
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    return round(float((z_a + z_b) * np.sqrt(2 * p * (1 - p) / n)), 4)
