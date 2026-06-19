import pandas as pd
import pytest
from analysis.pareto import find_pareto_frontier


def _df(*rows):
    return pd.DataFrame(rows)


def _make_row(variant, tps, mem, mmlu, model="llama", config="fp16"):
    return dict(variant=variant, model=model, config=config,
                tokens_per_sec_batch1=tps, memory_mb=mem, mmlu_accuracy=mmlu)


class TestFindParetoFrontier:

    def test_is_pareto_column_added(self):
        df = _df(_make_row("a", 100, 5000, 0.7))
        result = find_pareto_frontier(df)
        assert "is_pareto" in result.columns

    def test_single_point_always_pareto(self):
        df = _df(_make_row("a", 100, 5000, 0.7))
        result = find_pareto_frontier(df)
        assert result["is_pareto"].all()

    def test_dominated_point_excluded(self):
        # b is strictly worse on all three axes than a
        df = _df(
            _make_row("a", tps=100, mem=4000, mmlu=0.80),
            _make_row("b", tps=50,  mem=6000, mmlu=0.60),
        )
        result = find_pareto_frontier(df)
        is_pareto = result.set_index("variant")["is_pareto"]
        assert is_pareto["a"] == True
        assert is_pareto["b"] == False

    def test_incomparable_points_both_pareto(self):
        # a wins on tps; b wins on memory and mmlu — neither dominates
        df = _df(
            _make_row("a", tps=120, mem=8000, mmlu=0.65),
            _make_row("b", tps=60,  mem=3000, mmlu=0.80),
        )
        result = find_pareto_frontier(df)
        assert result["is_pareto"].all()

    def test_original_dataframe_unchanged(self):
        df = _df(_make_row("a", 100, 5000, 0.7))
        original_cols = list(df.columns)
        find_pareto_frontier(df)
        assert list(df.columns) == original_cols  # no side effects on input

    def test_pareto_count_realistic_scenario(self):
        # "slow_heavy" is strictly worse than int8 on all three axes → not Pareto.
        # fp16, int8, int4 are mutually incomparable → all Pareto.
        rows = [
            _make_row("fp16",       tps=120, mem=7000, mmlu=0.78),
            _make_row("int8",       tps=90,  mem=4500, mmlu=0.76),
            _make_row("int4",       tps=70,  mem=2500, mmlu=0.72),
            _make_row("slow_heavy", tps=65,  mem=5000, mmlu=0.71),  # dominated by int8
        ]
        result = find_pareto_frontier(_df(*rows))
        is_pareto = result.set_index("variant")["is_pareto"]
        assert is_pareto["fp16"]       == True
        assert is_pareto["int8"]       == True
        assert is_pareto["int4"]       == True
        assert is_pareto["slow_heavy"] == False
