import numpy as np
import pandas as pd
import pytest
import torch

from research.predictor import compare_formats, fit_sensitivity_predictor
from research.sensitivity import (
    layer_quantization_error,
    layer_weight_statistics,
    nf4_levels,
    quantize_dequantize_nf4,
    quantize_dequantize_rtn,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _heavy_tailed(shape, dfree):
    """Student-t style weights: lower dfree means heavier tails."""
    chi = torch.distributions.Chi2(dfree).sample(shape).sqrt() / dfree ** 0.5
    return torch.randn(*shape) / chi


class TestNF4Levels:
    def test_sixteen_distinct_sorted_levels(self):
        levels = nf4_levels()
        assert levels.numel() == 16
        assert bool((levels[1:] > levels[:-1]).all())

    def test_spans_normalized_range_and_includes_zero(self):
        levels = nf4_levels()
        assert float(levels[0]) == pytest.approx(-1.0)
        assert float(levels[-1]) == pytest.approx(1.0)
        assert bool((levels == 0).any())


class TestQuantizeDequantize:
    def test_shape_is_preserved(self):
        w = torch.randn(100, 37)
        assert quantize_dequantize_rtn(w).numel() == w.numel()
        assert quantize_dequantize_nf4(w).numel() == w.numel()

    def test_non_multiple_of_block_size_is_handled(self):
        # 37*3 = 111 elements, not divisible by the 64-element block
        w = torch.randn(3, 37)
        assert quantize_dequantize_nf4(w).numel() == 111

    def test_more_bits_reduce_error(self):
        w = torch.randn(512, 128)
        err8 = layer_quantization_error(w, "int8")["relative_frobenius_error"]
        err4 = layer_quantization_error(w, "fp4")["relative_frobenius_error"]
        assert err8 < err4

    def test_nf4_beats_fp4_on_gaussian_weights(self):
        # The QLoRA claim: normal-quantile levels suit normally-distributed weights
        w = torch.randn(2048, 256)
        err_fp4 = layer_quantization_error(w, "fp4")["relative_frobenius_error"]
        err_nf4 = layer_quantization_error(w, "nf4")["relative_frobenius_error"]
        assert err_nf4 < err_fp4

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            layer_quantization_error(torch.randn(8, 8), "int3")


class TestWeightStatistics:
    def test_gaussian_has_near_zero_excess_kurtosis(self):
        stats = layer_weight_statistics(torch.randn(4096, 64))
        assert abs(stats["kurtosis"]) < 0.5

    def test_heavy_tails_raise_kurtosis_and_block_range(self):
        light = layer_weight_statistics(torch.randn(2048, 128))
        heavy = layer_weight_statistics(_heavy_tailed((2048, 128), 3.0))
        assert heavy["kurtosis"] > light["kurtosis"]
        assert heavy["block_dynamic_range"] > light["block_dynamic_range"]

    def test_constant_weights_do_not_divide_by_zero(self):
        stats = layer_weight_statistics(torch.ones(256, 8))
        assert stats["kurtosis"] == 0.0
        assert np.isfinite(stats["outlier_ratio"])


class TestQuantizationErrorMechanism:
    def test_error_increases_with_tail_heaviness(self):
        """Core hypothesis: block absmax scaling means outliers coarsen the
        grid for their whole 64-element block, so heavy tails cost accuracy."""
        light = layer_quantization_error(torch.randn(2048, 128), "fp4")
        heavy = layer_quantization_error(_heavy_tailed((2048, 128), 2.5), "fp4")
        assert heavy["relative_frobenius_error"] > light["relative_frobenius_error"]

    def test_nf4_advantage_widens_with_heavier_tails(self):
        """Corollary: uniform FP4 wastes resolution on an empty range when
        weights are heavy-tailed, so NF4's edge should grow."""
        def advantage(dfree):
            w = _heavy_tailed((4096, 128), dfree)
            fp4 = layer_quantization_error(w, "fp4")["relative_frobenius_error"]
            nf4 = layer_quantization_error(w, "nf4")["relative_frobenius_error"]
            return (fp4 - nf4) / fp4

        assert advantage(2.5) > advantage(30.0)


def _synthetic_profile():
    rows = []
    for family, base in (("famA", 3.0), ("famB", 8.0)):
        for depth in range(12):
            dfree = base + depth * 0.7
            w = _heavy_tailed((1024, 256), dfree)
            stats = layer_weight_statistics(w)
            for method in ("fp4", "nf4", "int8"):
                rows.append({
                    "layer": f"{family}.l{depth}", "model_family": family,
                    "depth": depth, "layer_type": "proj",
                    **stats, **layer_quantization_error(w, method),
                })
    return pd.DataFrame(rows)


class TestPredictor:
    def test_cross_family_prediction_beats_mean_baseline(self):
        result = fit_sensitivity_predictor(_synthetic_profile(), "fp4")
        assert result["split"] == "leave-one-model_family-out"
        assert result["models"]["ridge"]["beats_baseline"] is True

    def test_reports_heldout_r2(self):
        result = fit_sensitivity_predictor(_synthetic_profile(), "fp4")
        assert "r2_heldout" in result["models"]["ridge"]
        assert "r2_baseline" in result["models"]["ridge"]

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="No rows for method"):
            fit_sensitivity_predictor(_synthetic_profile(), "int2")

    def test_format_comparison_links_advantage_to_tails(self):
        comparison = compare_formats(_synthetic_profile())
        assert comparison["nf4_relative_advantage"] > 0
        correlations = comparison["advantage_vs_tail_correlation"]
        assert correlations["block_dynamic_range"] > 0.5
