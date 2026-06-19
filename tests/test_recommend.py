import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from analysis.recommend import _minmax, generate_recommendation


def _make_df():
    return pd.DataFrame([
        dict(variant="llama_fp16",     model="llama", config="fp16",
             memory_mb=6500.0, tokens_per_sec_batch1=120.0, mmlu_accuracy=0.78,
             ttft_p50_ms=210.0, consistency_score=1.0, relevance_score=0.86),
        dict(variant="llama_int8",     model="llama", config="int8",
             memory_mb=4200.0, tokens_per_sec_batch1=90.0,  mmlu_accuracy=0.76,
             ttft_p50_ms=190.0, consistency_score=1.0, relevance_score=0.84),
        dict(variant="llama_int4",     model="llama", config="int4",
             memory_mb=2400.0, tokens_per_sec_batch1=70.0,  mmlu_accuracy=0.70,
             ttft_p50_ms=170.0, consistency_score=1.0, relevance_score=0.80),
        dict(variant="llama_int4_nf4", model="llama", config="int4_nf4",
             memory_mb=2500.0, tokens_per_sec_batch1=72.0,  mmlu_accuracy=0.72,
             ttft_p50_ms=172.0, consistency_score=1.0, relevance_score=0.81),
    ])


class TestMinmax:

    def test_min_maps_to_zero(self):
        s = pd.Series([0.0, 0.5, 1.0])
        assert _minmax(s)[0] == pytest.approx(0.0)

    def test_max_maps_to_one(self):
        s = pd.Series([0.0, 0.5, 1.0])
        assert _minmax(s)[2] == pytest.approx(1.0)

    def test_midpoint_maps_to_half(self):
        s = pd.Series([0.0, 0.5, 1.0])
        assert _minmax(s)[1] == pytest.approx(0.5)

    def test_constant_series_returns_half(self):
        s = pd.Series([7.0, 7.0, 7.0])
        assert (_minmax(s) == 0.5).all()


class TestGenerateRecommendation:

    def test_latency_priority_picks_highest_tps(self, tmp_path):
        df = _make_df()
        with patch("analysis.recommend._RECOMMENDATION_JSON", tmp_path / "rec.json"):
            rec = generate_recommendation(df)
        assert rec["latency_priority"]["variant"] == "llama_fp16"

    def test_memory_priority_picks_lowest_memory_above_threshold(self, tmp_path):
        df = _make_df()
        with patch("analysis.recommend._RECOMMENDATION_JSON", tmp_path / "rec.json"):
            rec = generate_recommendation(df)
        # All variants exceed 0.65; lowest memory is int4 at 2400 MB
        assert rec["memory_priority"]["variant"] == "llama_int4"

    def test_memory_priority_fallback_when_no_variant_exceeds_threshold(self, tmp_path):
        df = _make_df().copy()
        df["mmlu_accuracy"] = 0.50  # all below 0.65 threshold
        with patch("analysis.recommend._RECOMMENDATION_JSON", tmp_path / "rec.json"):
            rec = generate_recommendation(df)
        # Falls back to lowest memory regardless of quality
        assert rec["memory_priority"]["variant"] == "llama_int4"

    def test_recommendation_has_all_three_categories(self, tmp_path):
        df = _make_df()
        with patch("analysis.recommend._RECOMMENDATION_JSON", tmp_path / "rec.json"):
            rec = generate_recommendation(df)
        assert set(rec.keys()) == {"latency_priority", "memory_priority", "balanced"}

    def test_each_category_has_required_keys(self, tmp_path):
        required = {"variant", "model", "config", "memory_mb",
                    "tokens_per_sec_batch1", "mmlu_accuracy", "rationale"}
        df = _make_df()
        with patch("analysis.recommend._RECOMMENDATION_JSON", tmp_path / "rec.json"):
            rec = generate_recommendation(df)
        for category in rec.values():
            assert required.issubset(category.keys())

    def test_json_written_to_disk(self, tmp_path):
        out = tmp_path / "rec.json"
        df = _make_df()
        with patch("analysis.recommend._RECOMMENDATION_JSON", out):
            generate_recommendation(df)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "balanced" in data

    def test_rationale_is_non_empty_string(self, tmp_path):
        df = _make_df()
        with patch("analysis.recommend._RECOMMENDATION_JSON", tmp_path / "rec.json"):
            rec = generate_recommendation(df)
        for category in rec.values():
            assert isinstance(category["rationale"], str)
            assert len(category["rationale"]) > 0
