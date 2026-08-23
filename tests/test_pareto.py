import pandas as pd
import pytest

from analysis.pareto import find_pareto_frontier


def _row(variant, tps, mem, mmlu, model="llama", config="fp16"):
    return {
        "variant": variant, "model": model, "config": config,
        "tokens_per_sec_batch1": tps, "memory_mb": mem, "mmlu_accuracy": mmlu,
    }


def _frontier(*rows, use_epsilon=False):
    result = find_pareto_frontier(pd.DataFrame(rows), use_epsilon=use_epsilon)
    return result.set_index("variant")["is_pareto"]


class TestFindParetoFrontier:
    def test_adds_is_pareto_column(self):
        result = find_pareto_frontier(pd.DataFrame([_row("a", 100, 5000, 0.7)]))
        assert "is_pareto" in result.columns

    def test_single_point_is_pareto(self):
        assert _frontier(_row("a", 100, 5000, 0.7))["a"]

    def test_strictly_dominated_point_excluded(self):
        flags = _frontier(
            _row("a", 100, 4000, 0.80),
            _row("b", 50, 6000, 0.60),
        )
        assert flags["a"]
        assert not flags["b"]

    def test_incomparable_points_both_kept(self):
        # a wins throughput; b wins memory and accuracy
        flags = _frontier(
            _row("a", 200, 6000, 0.60),
            _row("b", 80, 2000, 0.75),
        )
        assert flags["a"] and flags["b"]

    def test_memory_is_minimized_not_maximized(self):
        # identical except memory; the smaller one must win
        flags = _frontier(
            _row("small", 100, 2000, 0.70),
            _row("big", 100, 8000, 0.70),
        )
        assert flags["small"]
        assert not flags["big"]

    def test_missing_objective_column_raises(self):
        df = pd.DataFrame([{"variant": "a", "memory_mb": 1.0, "mmlu_accuracy": 0.5}])
        with pytest.raises(KeyError, match="Missing objective"):
            find_pareto_frontier(df)


class TestEpsilonDominance:
    def test_sub_noise_advantage_does_not_dominate(self):
        """A 0.001 accuracy edge is measurement noise, not superiority.
        Without epsilon, noise decides which config is called optimal."""
        rows = (_row("a", 100.1, 4000, 0.701), _row("b", 100.0, 4000, 0.700))
        strict = _frontier(*rows, use_epsilon=False)
        tolerant = _frontier(*rows, use_epsilon=True)
        assert not strict["b"]      # naive frontier drops b on noise
        assert tolerant["b"]        # epsilon frontier keeps both

    def test_real_advantage_still_dominates_under_epsilon(self):
        flags = _frontier(
            _row("a", 200, 2000, 0.80),
            _row("b", 50, 8000, 0.40),
            use_epsilon=True,
        )
        assert flags["a"]
        assert not flags["b"]
