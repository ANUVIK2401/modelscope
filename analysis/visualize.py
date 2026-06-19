from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # headless-safe; no display required on Colab

_PLOTS_DIR = Path("results/plots")
_DPI = 150

_MODEL_COLORS: dict[str, str] = {
    "llama": "#4C72B0",
    "gemma": "#DD8452",
}
_CONFIG_MARKERS: dict[str, str] = {
    "fp16":     "o",
    "int8":     "s",
    "int4":     "^",
    "int4_nf4": "D",
}


def _scatter_with_pareto(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    for (model_name, config_name), group in df.groupby(["model", "config"]):
        color = _MODEL_COLORS.get(model_name, "gray")
        marker = _CONFIG_MARKERS.get(config_name, "o")
        ax.scatter(
            group[x_col], group[y_col],
            color=color, marker=marker, s=90, zorder=3,
            label=f"{model_name}/{config_name}",
        )

    # Gold star ring on top of every Pareto-optimal point.
    pareto = df[df["is_pareto"]]
    if not pareto.empty:
        ax.scatter(
            pareto[x_col], pareto[y_col],
            marker="*", s=320,
            facecolors="none", edgecolors="goldenrod", linewidths=1.8,
            zorder=4, label="Pareto frontier",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)


def plot_memory_vs_throughput(df: pd.DataFrame) -> None:
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    _scatter_with_pareto(
        ax, df,
        x_col="memory_mb",
        y_col="tokens_per_sec_batch1",
        x_label="Peak memory (MB)",
        y_label="Throughput — batch 1 (tok/s)",
        title="Memory vs Throughput",
    )
    fig.tight_layout()
    fig.savefig(_PLOTS_DIR / "memory_vs_throughput.png", dpi=_DPI)
    plt.close(fig)


def plot_quality_vs_speed(df: pd.DataFrame) -> None:
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    _scatter_with_pareto(
        ax, df,
        x_col="tokens_per_sec_batch1",
        y_col="mmlu_accuracy",
        x_label="Throughput — batch 1 (tok/s)",
        y_label="MMLU accuracy",
        title="Quality vs Speed",
    )
    fig.tight_layout()
    fig.savefig(_PLOTS_DIR / "quality_vs_speed.png", dpi=_DPI)
    plt.close(fig)


def plot_pareto_frontier(df: pd.DataFrame) -> None:
    """
    2D bubble chart: x=memory, y=throughput, bubble area ∝ mmlu_accuracy.
    Encodes three axes in one 2-D view. Pareto points get a gold border.
    """
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    # Scale bubble area so the largest accuracy gives a diameter ~30pts.
    max_acc = df["mmlu_accuracy"].max() if df["mmlu_accuracy"].max() > 0 else 1.0
    sizes = (df["mmlu_accuracy"] / max_acc) * 600 + 60  # floor so zero-accuracy still shows

    colors = [_MODEL_COLORS.get(m, "gray") for m in df["model"]]
    edge_colors = ["goldenrod" if p else "k" for p in df["is_pareto"]]
    linewidths = [2.0 if p else 0.5 for p in df["is_pareto"]]

    ax.scatter(
        df["memory_mb"], df["tokens_per_sec_batch1"],
        s=sizes, c=colors, edgecolors=edge_colors, linewidths=linewidths,
        alpha=0.80, zorder=3,
    )

    # Annotate each bubble with its variant key.
    for _, row in df.iterrows():
        ax.annotate(
            row["variant"],
            xy=(row["memory_mb"], row["tokens_per_sec_batch1"]),
            fontsize=7, ha="center", va="bottom",
            xytext=(0, 6), textcoords="offset points",
        )

    # Manual legend: model colors + Pareto indicator.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=9, label=m)
        for m, c in _MODEL_COLORS.items()
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="goldenrod", markeredgewidth=2,
               markersize=10, label="Pareto frontier"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="best")

    ax.set_xlabel("Peak memory (MB)")
    ax.set_ylabel("Throughput — batch 1 (tok/s)")
    ax.set_title("Pareto Frontier  ·  bubble area = MMLU accuracy  ·  gold border = Pareto-optimal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_PLOTS_DIR / "pareto_frontier.png", dpi=_DPI)
    plt.close(fig)
