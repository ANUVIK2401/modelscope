"""Pairwise significance testing across variants.

This module is what stands between a measured difference and a stated finding.
Every accuracy comparison in the writeup is generated from here, so a claim that
does not clear its confidence interval cannot reach the README.

Multiple-comparison correction matters: 8 variants give 28 pairs, and at
alpha=0.05 roughly 1.4 spurious "significant" results are expected by chance
alone. Holm-Bonferroni controls the family-wise error rate without the
conservatism of plain Bonferroni.
"""

import itertools
import json
from typing import Any

import numpy as np

from analysis.paths import SIGNIFICANCE_JSON
from analysis.statistics import (
    accuracy_ci,
    mcnemar_test,
    minimum_detectable_effect,
    paired_bootstrap_difference,
)


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down. Returns per-hypothesis rejection flags."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    rejected = [False] * n
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            break  # step-down: stop at first non-rejection
    return rejected


def compare_all_pairs(
    flags_by_variant: dict[str, list[int]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired comparison of every variant pair on the shared question set."""
    variants = sorted(flags_by_variant)
    lengths = {len(v) for v in flags_by_variant.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"Variants scored on different question counts {lengths}. "
            "Paired tests require the identical question set for every variant."
        )
    n_questions = lengths.pop() if lengths else 0

    comparisons: list[dict[str, Any]] = []
    for a, b in itertools.combinations(variants, 2):
        boot = paired_bootstrap_difference(flags_by_variant[a], flags_by_variant[b])
        mcn = mcnemar_test(flags_by_variant[a], flags_by_variant[b])
        comparisons.append({
            "variant_a": a,
            "variant_b": b,
            "accuracy_a": round(float(np.mean(flags_by_variant[a])), 4),
            "accuracy_b": round(float(np.mean(flags_by_variant[b])), 4),
            "difference": boot["difference"],
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "p_bootstrap": boot["p_value"],
            "p_mcnemar": mcn["p_value"],
            "significant_uncorrected": boot["significant"],
        })

    corrected = holm_bonferroni([c["p_mcnemar"] for c in comparisons], alpha)
    for comparison, flag in zip(comparisons, corrected, strict=True):
        comparison["significant_holm"] = bool(flag)

    per_variant = {
        name: dict(zip(("accuracy", "ci_low", "ci_high"),
                       [round(x, 4) for x in accuracy_ci(flags)], strict=True))
        for name, flags in flags_by_variant.items()
    }

    return {
        "n_questions": n_questions,
        "alpha": alpha,
        "minimum_detectable_effect": minimum_detectable_effect(n_questions),
        "n_comparisons": len(comparisons),
        "per_variant": per_variant,
        "comparisons": comparisons,
        "n_significant_holm": sum(corrected),
    }


def summarize_findings(report: dict[str, Any]) -> list[str]:
    """Plain-language sentences for only the claims that survived correction.

    Anything not significant after Holm is reported as indistinguishable, which
    is a result in its own right and the honest way to state a null.
    """
    lines: list[str] = []
    mde = report["minimum_detectable_effect"]
    lines.append(
        f"n={report['n_questions']} questions; smallest detectable accuracy gap "
        f"at 80% power is {mde:.3f} ({mde * 100:.1f} points). Differences below "
        "this cannot be resolved by this study."
    )
    for c in report["comparisons"]:
        gap = c["difference"]
        if c["significant_holm"]:
            better, worse = (
                (c["variant_a"], c["variant_b"]) if gap > 0
                else (c["variant_b"], c["variant_a"])
            )
            lines.append(
                f"{better} > {worse} by {abs(gap):.3f} "
                f"(95% CI [{c['ci_low']:.3f}, {c['ci_high']:.3f}], "
                f"McNemar p={c['p_mcnemar']:.4f}, significant after Holm)."
            )
        else:
            lines.append(
                f"{c['variant_a']} vs {c['variant_b']}: difference "
                f"{gap:+.3f}, CI [{c['ci_low']:.3f}, {c['ci_high']:.3f}] "
                "includes zero - statistically indistinguishable."
            )
    return lines


def run(flags_path=None, output_path=SIGNIFICANCE_JSON) -> dict[str, Any]:
    from eval.harness import PER_QUESTION_JSON

    path = flags_path or PER_QUESTION_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run eval/harness.py first to produce "
            "per-question correctness vectors."
        )
    flags = json.loads(path.read_text())
    report = compare_all_pairs(flags)
    report["findings"] = summarize_findings(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    result = run()
    for line in result["findings"]:
        print(line)
