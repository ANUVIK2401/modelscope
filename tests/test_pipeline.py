"""End-to-end pipeline test on synthetic results.

Exercises merge -> Pareto -> significance -> recommendation -> figures without a
GPU, so a broken analysis chain is caught in CI rather than after a two-hour
Colab run.
"""

import csv
import json

import numpy as np
import pandas as pd
import pytest

from analysis.pareto import find_pareto_frontier, load_results
from analysis.recommend import generate_recommendation
from analysis.significance import compare_all_pairs
from analysis.visualize import build_summary_figure

VARIANTS = [
    # variant, model, config, memory, tok/s, accuracy
    ("llama_fp16", "llama", "fp16", 6128.0, 22.3, 0.63),
    ("llama_int8", "llama", "int8", 3515.0, 6.9, 0.62),
    ("llama_fp4", "llama", "fp4", 2299.0, 13.8, 0.55),
    ("llama_nf4", "llama", "nf4", 2299.0, 14.7, 0.60),
    ("gemma_fp16", "gemma", "fp16", 4980.0, 19.5, 0.57),
    ("gemma_int8", "gemma", "int8", 2890.0, 5.5, 0.56),
    ("gemma_fp4", "gemma", "fp4", 1870.0, 13.4, 0.48),
    ("gemma_nf4", "gemma", "nf4", 1870.0, 13.3, 0.53),
]


def _write_csvs(tmp_path):
    bench_path = tmp_path / "benchmark_results.csv"
    qual_path = tmp_path / "quality_results.csv"

    bench_rows = []
    qual_rows = []
    for variant, model, config, memory, tps, accuracy in VARIANTS:
        bench_rows.append({
            "variant": variant, "model": model, "config": config,
            "status": "ok", "error": "", "env_hash": "abc123",
            "weight_memory_mb": memory, "memory_mb": memory,
            "tokens_per_sec_batch1": tps, "ttft_p50_ms": 1000 / tps * 2,
            "itl_p50_ms": 1000 / tps, "load_time_s": 30.0,
        })
        qual_rows.append({
            "variant": variant, "model": model, "config": config,
            "status": "ok", "error": "", "env_hash": "abc123",
            "mmlu_accuracy": accuracy,
            "mmlu_ci_low": round(accuracy - 0.055, 4),
            "mmlu_ci_high": round(accuracy + 0.055, 4),
            "mmlu_n": 320, "ece": 0.08,
            "mmlu_reasoning": round(accuracy - 0.06, 4),
            "mmlu_recall": round(accuracy + 0.04, 4),
            "mmlu_applied": accuracy,
            "consistency_score": 0.58, "consistency_std": 0.05,
            "perplexity": 9.0 + (1 - accuracy) * 8,
        })

    for path, rows in ((bench_path, bench_rows), (qual_path, qual_rows)):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return bench_path, qual_path


@pytest.fixture
def results(tmp_path):
    bench_path, qual_path = _write_csvs(tmp_path)
    return load_results(bench_path, qual_path)


class TestLoadResults:
    def test_merges_all_variants(self, results):
        assert len(results) == len(VARIANTS)

    def test_failed_rows_are_dropped(self, tmp_path):
        bench_path, qual_path = _write_csvs(tmp_path)
        frame = pd.read_csv(bench_path)
        frame.loc[frame["variant"] == "gemma_fp4", "status"] = "failed"
        frame.to_csv(bench_path, index=False)
        merged = load_results(bench_path, qual_path)
        assert "gemma_fp4" not in set(merged["variant"])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nope.csv", tmp_path / "also_nope.csv")


class TestPipeline:
    def test_pareto_frontier_is_non_empty_and_partial(self, results):
        frontier = find_pareto_frontier(results)
        n_pareto = int(frontier["is_pareto"].sum())
        assert 0 < n_pareto < len(frontier)

    def test_fp16_is_pareto_optimal_on_accuracy(self, results):
        """The most accurate config cannot be dominated on the accuracy axis."""
        frontier = find_pareto_frontier(results)
        best = frontier.loc[frontier["mmlu_accuracy"].idxmax()]
        assert bool(best["is_pareto"])

    def test_recommendation_generates_from_real_schema(self, results, tmp_path):
        rec = generate_recommendation(results, tmp_path / "rec.json")
        assert rec["latency_priority"]["variant"] == "llama_fp16"
        assert json.loads((tmp_path / "rec.json").read_text())["caveat"]

    def test_nf4_preferred_over_fp4_at_equal_memory(self, results, tmp_path):
        """nf4 and fp4 tie on memory; nf4 has the smaller accuracy drop, so the
        memory recommendation should prefer it."""
        rec = generate_recommendation(results, tmp_path / "rec.json")
        assert rec["memory_priority"]["config"] == "nf4"

    def test_significance_report_covers_all_pairs(self):
        rng = np.random.default_rng(1)
        flags = {
            variant: (rng.random(320) < accuracy).astype(int).tolist()
            for variant, _m, _c, _mem, _tps, accuracy in VARIANTS
        }
        report = compare_all_pairs(flags)
        assert report["n_comparisons"] == 28
        assert report["n_questions"] == 320

    def test_summary_figure_is_written(self, results, tmp_path, monkeypatch):
        import analysis.visualize as visualize
        monkeypatch.setattr(visualize, "PLOTS_DIR", tmp_path)
        output = build_summary_figure(results)
        assert output.exists()
        assert output.stat().st_size > 10_000


class TestReportGeneration:
    def test_ci_sign_matches_stated_direction(self):
        """Regression: the CI is computed on (a - b). When b is the better
        variant the sentence says 'b beats a', so the interval must be negated
        and re-ordered or it prints a negative range for a positive gap."""
        from scripts.build_report import significance_section

        report = {
            "n_questions": 320, "minimum_detectable_effect": 0.111,
            "n_significant_holm": 1, "n_comparisons": 1,
            "comparisons": [{
                "variant_a": "gemma_fp4", "variant_b": "llama_fp16",
                "difference": -0.138, "ci_low": -0.212, "ci_high": -0.062,
                "p_mcnemar": 0.0005, "significant_holm": True,
            }],
        }
        text = significance_section(report)
        assert "`llama_fp16` beats `gemma_fp4`" in text
        assert "[0.062, 0.212]" in text
        assert "[-0.212" not in text

    def test_null_result_is_stated_as_a_finding(self):
        from scripts.build_report import significance_section

        report = {
            "n_questions": 320, "minimum_detectable_effect": 0.111,
            "n_significant_holm": 0, "n_comparisons": 28, "comparisons": [],
        }
        assert "indistinguishable" in significance_section(report)
