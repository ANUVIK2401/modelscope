import json

import pandas as pd
import pytest

from analysis.recommend import generate_recommendation


def _df():
    return pd.DataFrame([
        dict(variant="llama_fp16", model="llama", config="fp16",
             memory_mb=6500.0, tokens_per_sec_batch1=22.0, mmlu_accuracy=0.63,
             mmlu_ci_low=0.57, mmlu_ci_high=0.69, ttft_p50_ms=210.0,
             itl_p50_ms=45.0, consistency_score=0.62, perplexity=9.1, ece=0.06),
        dict(variant="llama_int8", model="llama", config="int8",
             memory_mb=3500.0, tokens_per_sec_batch1=7.0, mmlu_accuracy=0.62,
             mmlu_ci_low=0.56, mmlu_ci_high=0.68, ttft_p50_ms=310.0,
             itl_p50_ms=142.0, consistency_score=0.60, perplexity=9.3, ece=0.07),
        dict(variant="llama_fp4", model="llama", config="fp4",
             memory_mb=2300.0, tokens_per_sec_batch1=14.0, mmlu_accuracy=0.55,
             mmlu_ci_low=0.49, mmlu_ci_high=0.61, ttft_p50_ms=250.0,
             itl_p50_ms=71.0, consistency_score=0.55, perplexity=11.4, ece=0.11),
        dict(variant="llama_nf4", model="llama", config="nf4",
             memory_mb=2300.0, tokens_per_sec_batch1=14.5, mmlu_accuracy=0.60,
             mmlu_ci_low=0.54, mmlu_ci_high=0.66, ttft_p50_ms=245.0,
             itl_p50_ms=69.0, consistency_score=0.58, perplexity=10.1, ece=0.08),
    ])


class TestGenerateRecommendation:
    def test_latency_priority_picks_fastest(self, tmp_path):
        rec = generate_recommendation(_df(), tmp_path / "rec.json")
        assert rec["latency_priority"]["variant"] == "llama_fp16"

    def test_memory_priority_respects_quality_floor(self, tmp_path):
        """fp4 is smallest but drops 0.08 below the fp16 baseline (> 0.05
        floor); nf4 ties on memory and stays within 0.03, so it wins."""
        rec = generate_recommendation(_df(), tmp_path / "rec.json")
        assert rec["memory_priority"]["variant"] == "llama_nf4"

    def test_quality_floor_relaxes_when_nothing_qualifies(self, tmp_path):
        df = _df()
        df.loc[df["config"] != "fp16", "mmlu_accuracy"] = 0.20
        rec = generate_recommendation(df, tmp_path / "rec.json")
        assert "relaxed" in rec["memory_priority"]["rationale"]

    def test_all_three_categories_present(self, tmp_path):
        rec = generate_recommendation(_df(), tmp_path / "rec.json")
        assert {"latency_priority", "memory_priority", "balanced"} <= set(rec)

    def test_recommendations_report_confidence_intervals(self, tmp_path):
        """A point estimate without an interval invites over-reading a gap
        that the study is not powered to resolve."""
        rec = generate_recommendation(_df(), tmp_path / "rec.json")
        for key in ("latency_priority", "memory_priority", "balanced"):
            assert "mmlu_ci_low" in rec[key]
            assert "[" in rec[key]["rationale"]

    def test_no_relevance_score_dependency(self, tmp_path):
        """Regression: the previous version required a relevance_score column
        that no module ever produced, so it raised on real data."""
        assert "relevance_score" not in _df().columns
        generate_recommendation(_df(), tmp_path / "rec.json")  # must not raise

    def test_writes_json_with_caveat(self, tmp_path):
        out = tmp_path / "rec.json"
        generate_recommendation(_df(), out)
        data = json.loads(out.read_text())
        assert "caveat" in data
        assert "quality_floor" in data

    def test_empty_frame_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            generate_recommendation(pd.DataFrame(), tmp_path / "rec.json")
