import numpy as np
import pytest

from analysis.significance import compare_all_pairs, holm_bonferroni, summarize_findings


class TestHolmBonferroni:
    def test_all_null_p_values_rejected_none(self):
        assert holm_bonferroni([0.9, 0.8, 0.7]) == [False, False, False]

    def test_smallest_p_uses_strictest_threshold(self):
        # 0.02 > 0.05/3 = 0.0167, so nothing survives despite p < 0.05
        assert holm_bonferroni([0.02, 0.6, 0.7]) == [False, False, False]

    def test_clearly_significant_result_survives(self):
        assert holm_bonferroni([0.0001, 0.6, 0.7])[0] is True

    def test_is_more_conservative_than_uncorrected(self):
        """28 pairs at alpha=0.05 yields ~1.4 false positives by chance;
        correction is what keeps those out of the findings list."""
        p_values = [0.04] * 28
        assert sum(holm_bonferroni(p_values)) == 0

    def test_empty_input(self):
        assert holm_bonferroni([]) == []


class TestCompareAllPairs:
    def test_pair_count_is_n_choose_2(self):
        rng = np.random.default_rng(0)
        flags = {f"v{i}": rng.integers(0, 2, 60).tolist() for i in range(4)}
        assert compare_all_pairs(flags)["n_comparisons"] == 6

    def test_identical_variants_are_not_significant(self):
        shared = [1, 0, 1, 1, 0] * 12
        report = compare_all_pairs({"a": shared, "b": list(shared)})
        assert report["comparisons"][0]["significant_holm"] is False

    def test_large_gap_is_significant(self):
        report = compare_all_pairs({
            "good": [1] * 80 + [0] * 20,
            "bad": [0] * 80 + [1] * 20,
        })
        assert report["comparisons"][0]["significant_holm"] is True

    def test_mismatched_question_counts_raise(self):
        """Paired tests are only valid on a shared question set; silently
        comparing different sets would invalidate every p-value."""
        with pytest.raises(ValueError, match="different question counts"):
            compare_all_pairs({"a": [1, 0, 1], "b": [1, 0]})

    def test_reports_minimum_detectable_effect(self):
        flags = {"a": [1, 0] * 50, "b": [1, 1] + [0] * 98}
        assert compare_all_pairs(flags)["minimum_detectable_effect"] > 0

    def test_per_variant_intervals_present(self):
        report = compare_all_pairs({"a": [1, 0] * 50, "b": [1] * 100})
        assert "ci_low" in report["per_variant"]["a"]


class TestSummarizeFindings:
    def test_states_power_limit_first(self):
        report = compare_all_pairs({"a": [1, 0] * 50, "b": [0, 1] * 50})
        assert "detectable" in summarize_findings(report)[0]

    def test_null_result_is_reported_as_indistinguishable(self):
        shared = [1, 0, 1, 1, 0] * 12
        lines = summarize_findings(compare_all_pairs({"a": shared, "b": list(shared)}))
        assert any("indistinguishable" in line for line in lines)
