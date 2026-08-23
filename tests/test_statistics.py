import numpy as np
import pytest

from analysis.statistics import (
    accuracy_ci,
    bootstrap_ci,
    mcnemar_test,
    minimum_detectable_effect,
    paired_bootstrap_difference,
    summarize_runs,
)


class TestAccuracyCI:
    def test_wilson_matches_reference_value(self):
        # p=0.6, n=100 -> Wilson 95% CI approx (0.502, 0.691)
        p, lo, hi = accuracy_ci([1] * 60 + [0] * 40)
        assert p == pytest.approx(0.60)
        assert lo == pytest.approx(0.502, abs=0.002)
        assert hi == pytest.approx(0.691, abs=0.002)

    def test_interval_stays_within_unit_range(self):
        # Wilson's advantage over the normal approximation: no impossible bounds
        _p, lo, hi = accuracy_ci([1] * 20)
        assert lo >= 0.0
        assert hi <= 1.0

    def test_empty_input_returns_zeros(self):
        assert accuracy_ci([]) == (0.0, 0.0, 0.0)

    def test_interval_narrows_as_n_grows(self):
        _p, lo_small, hi_small = accuracy_ci([1, 0] * 25)
        _p, lo_large, hi_large = accuracy_ci([1, 0] * 500)
        assert (hi_large - lo_large) < (hi_small - lo_small)


class TestPairedBootstrap:
    def test_identical_vectors_give_zero_difference(self):
        flags = [1, 0, 1, 1, 0] * 20
        result = paired_bootstrap_difference(flags, flags)
        assert result["difference"] == 0.0
        assert result["significant"] is False

    def test_large_real_gap_is_significant(self):
        a = [1] * 90 + [0] * 10
        b = [0] * 90 + [1] * 10
        result = paired_bootstrap_difference(a, b)
        assert result["significant"] is True
        assert result["difference"] > 0

    def test_two_point_gap_at_n100_is_not_significant(self):
        # The README claimed int4 (0.37) beat int8 (0.35) on quality.
        # At n=100 that gap is far inside the noise floor.
        rng = np.random.default_rng(0)
        a = (rng.random(100) < 0.37).astype(int).tolist()
        b = (rng.random(100) < 0.35).astype(int).tolist()
        result = paired_bootstrap_difference(a, b)
        assert result["significant"] is False
        assert result["ci_low"] < 0 < result["ci_high"]

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="equal-length"):
            paired_bootstrap_difference([1, 0, 1], [1, 0])


class TestMcNemar:
    def test_only_discordant_pairs_matter(self):
        # Same 5 agreements added to both; verdict must not change
        a1, b1 = [1, 0, 1, 0], [0, 1, 0, 1]
        a2, b2 = a1 + [1] * 5, b1 + [1] * 5
        assert mcnemar_test(a1, b1)["p_value"] == mcnemar_test(a2, b2)["p_value"]

    def test_perfect_agreement_is_not_significant(self):
        flags = [1, 0, 1, 1]
        result = mcnemar_test(flags, flags)
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_one_sided_disagreement_is_significant(self):
        a = [1] * 20 + [0] * 5
        b = [0] * 20 + [0] * 5
        assert mcnemar_test(a, b)["significant"] is True


class TestMinimumDetectableEffect:
    def test_mde_at_n100_exceeds_readme_claimed_gaps(self):
        # The README compared configs differing by 0.02 accuracy at n=100.
        # The MDE proves that comparison was unpowered by an order of magnitude.
        mde = minimum_detectable_effect(100)
        assert mde > 0.15
        assert mde > 0.02 * 5

    def test_mde_shrinks_with_more_questions(self):
        assert minimum_detectable_effect(1000) < minimum_detectable_effect(100)


class TestBootstrapAndSummary:
    def test_ci_contains_mean(self):
        mean, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        assert lo <= mean <= hi

    def test_single_value_has_degenerate_interval(self):
        mean, lo, hi = bootstrap_ci([7.0])
        assert mean == lo == hi == 7.0

    def test_summarize_reports_n_and_percentiles(self):
        summary = summarize_runs([10.0, 20.0, 30.0, 40.0])
        assert summary["n"] == 4
        assert summary["p50"] == pytest.approx(25.0)
        assert summary["p95"] <= 40.0
