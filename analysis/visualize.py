"""Publication-format figures.

Design rules: every accuracy point carries its confidence interval, no chart
implies precision the measurement does not have, and figures are legible in
grayscale (marker shape carries config, not colour alone).
"""

from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.paths import PLOTS_DIR  # noqa: E402

DPI = 200
MODEL_COLORS = {"llama": "#3B6EA5", "gemma": "#C1743C"}
CONFIG_MARKERS = {"fp16": "o", "int8": "s", "fp4": "^", "nf4": "D"}

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _style(row: pd.Series) -> dict[str, Any]:
    return {
        "color": MODEL_COLORS.get(row["model"], "gray"),
        "marker": CONFIG_MARKERS.get(row["config"], "o"),
    }


def _annotate(ax, df: pd.DataFrame, x: str, y: str) -> None:
    for _, row in df.iterrows():
        ax.annotate(row["variant"], (row[x], row[y]), fontsize=6.5,
                    xytext=(0, 7), textcoords="offset points", ha="center")


def plot_accuracy_with_intervals(df: pd.DataFrame, ax=None):
    """Accuracy per variant with CIs and the random-guess floor marked.

    The chance line is the single most important annotation: it is what makes
    an at-or-below-chance result visually obvious instead of a plausible bar.
    """
    own_figure = ax is None
    if own_figure:
        _fig, ax = plt.subplots(figsize=(7, 4))

    ordered = df.sort_values("mmlu_accuracy", ascending=False).reset_index(drop=True)
    positions = range(len(ordered))

    has_ci = "mmlu_ci_low" in ordered.columns
    errors = None
    if has_ci:
        errors = [
            (ordered["mmlu_accuracy"] - ordered["mmlu_ci_low"]).clip(lower=0),
            (ordered["mmlu_ci_high"] - ordered["mmlu_accuracy"]).clip(lower=0),
        ]

    ax.bar(positions, ordered["mmlu_accuracy"],
           color=[MODEL_COLORS.get(m, "gray") for m in ordered["model"]],
           yerr=errors, capsize=3, alpha=0.85, edgecolor="black", linewidth=0.4)

    ax.axhline(0.25, ls="--", lw=1.2, color="crimson")
    ax.text(len(ordered) - 0.5, 0.255, "random guess (0.25)",
            fontsize=7, color="crimson", ha="right")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(ordered["variant"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("MMLU accuracy (5-shot)")
    ax.set_title("Accuracy with 95% confidence intervals")
    ax.set_ylim(0, max(0.8, float(ordered["mmlu_accuracy"].max()) * 1.25))

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=MODEL_COLORS[m], edgecolor="black", label=m)
                 for m in sorted(ordered["model"].unique())
                 if m in MODEL_COLORS],
        fontsize=7, title="model", title_fontsize=7, loc="upper right",
    )

    if own_figure:
        plt.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "accuracy_intervals.png", dpi=DPI)
        plt.close()
    return ax


def plot_memory_vs_throughput(df: pd.DataFrame, ax=None):
    own_figure = ax is None
    if own_figure:
        _fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for _, row in df.iterrows():
        ax.scatter(row["memory_mb"], row["tokens_per_sec_batch1"],
                   s=95, zorder=3, edgecolor="black", linewidth=0.5, **_style(row))
        if row.get("is_pareto", False):
            ax.scatter(row["memory_mb"], row["tokens_per_sec_batch1"],
                       s=320, marker="*", facecolors="none",
                       edgecolors="goldenrod", linewidths=1.6, zorder=4)

    _annotate(ax, df, "memory_mb", "tokens_per_sec_batch1")
    ax.set_xlabel("Weight memory (MB)")
    ax.set_ylabel("Throughput, batch 1 (tok/s)")
    ax.set_title("Memory vs throughput (gold star = Pareto-optimal)")

    if own_figure:
        plt.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "memory_vs_throughput.png", dpi=DPI)
        plt.close()
    return ax


def plot_category_degradation(df: pd.DataFrame, ax=None):
    """Per-capability accuracy drop relative to each family's FP16 baseline.

    Aggregate MMLU hides the effect that matters: if quantization costs more on
    reasoning than on recall, a single averaged number cannot show it.
    """
    own_figure = ax is None
    if own_figure:
        _fig, ax = plt.subplots(figsize=(7, 4))

    category_columns = [c for c in df.columns if c.startswith("mmlu_")
                        and c.split("mmlu_")[-1] in ("reasoning", "recall", "applied")]
    if not category_columns:
        ax.text(0.5, 0.5, "No per-category data available",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    baselines = df[df["config"] == "fp16"].set_index("model")
    quantized = df[df["config"] != "fp16"]

    width = 0.8 / max(len(category_columns), 1)
    positions = range(len(quantized))

    for i, column in enumerate(category_columns):
        drops = []
        for _, row in quantized.iterrows():
            base = baselines.loc[row["model"], column]
            drops.append(float(base) - float(row[column]))
        offsets = [p + i * width - 0.4 + width / 2 for p in positions]
        ax.bar(offsets, drops, width=width,
               label=column.replace("mmlu_", ""), alpha=0.88,
               edgecolor="black", linewidth=0.4)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(quantized["variant"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy drop vs FP16")
    ax.set_title("Task-conditional degradation (higher = worse)")
    ax.legend(fontsize=7, title="capability", title_fontsize=7)

    if own_figure:
        plt.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "category_degradation.png", dpi=DPI)
        plt.close()
    return ax


def plot_sensitivity_scatter(profile: pd.DataFrame, ax=None):
    """Layer quantization error against block dynamic range.

    This is the mechanism figure: bitsandbytes shares one absmax scale across
    each 64-weight block, so a block containing an outlier gets a coarser
    effective grid for every weight in it.
    """
    own_figure = ax is None
    if own_figure:
        _fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for method, marker in (("fp4", "^"), ("nf4", "D")):
        subset = profile[profile["method"] == method]
        if subset.empty:
            continue
        ax.scatter(subset["block_dynamic_range"], subset["relative_frobenius_error"],
                   s=22, alpha=0.6, marker=marker, label=method,
                   edgecolor="black", linewidth=0.25)

    ax.set_xlabel("Block dynamic range (mean per-block max/mean |w|)")
    ax.set_ylabel("Relative Frobenius reconstruction error")
    ax.set_title("Layer sensitivity vs weight-distribution shape")
    ax.legend(fontsize=8, title="format", title_fontsize=8)

    if own_figure:
        plt.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "layer_sensitivity.png", dpi=DPI)
        plt.close()
    return ax


def build_summary_figure(df: pd.DataFrame, profile: pd.DataFrame | None = None):
    """Four-panel summary figure for the writeup."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_memory_vs_throughput(df, axes[0][0])
    plot_accuracy_with_intervals(df, axes[0][1])
    plot_category_degradation(df, axes[1][0])

    if profile is not None and not profile.empty:
        plot_sensitivity_scatter(profile, axes[1][1])
    else:
        axes[1][1].text(0.5, 0.5, "Run research/run_sensitivity.py",
                        ha="center", va="center", transform=axes[1][1].transAxes)

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output = PLOTS_DIR / "final_analysis.png"
    fig.savefig(output, dpi=DPI)
    plt.close(fig)
    return output
