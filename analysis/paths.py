"""Canonical output paths, resolved relative to the project root.

The previous runners hardcoded "/content/modelscope/results", which made them
unrunnable outside Colab and violated the project's own path rule. Root is
derived from this file's location, with MODELSCOPE_ROOT as an override for
environments that relocate the tree.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("MODELSCOPE_ROOT", Path(__file__).resolve().parent.parent)
)

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

BENCHMARK_CSV = RESULTS_DIR / "benchmark_results.csv"
QUALITY_CSV = RESULTS_DIR / "quality_results.csv"
SENSITIVITY_CSV = RESULTS_DIR / "layer_sensitivity.csv"
PREDICTION_CSV = RESULTS_DIR / "degradation_predictions.csv"
MERGED_CSV = RESULTS_DIR / "merged_results.csv"
RECOMMENDATION_JSON = RESULTS_DIR / "recommendation.json"
SIGNIFICANCE_JSON = RESULTS_DIR / "significance_tests.json"
ENV_JSON = RESULTS_DIR / "environment.json"


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
