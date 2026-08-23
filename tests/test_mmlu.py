"""Tests for MMLU scoring, including regressions for the parser bug that
produced below-chance accuracy in the previous implementation."""

import pytest
import torch

from eval.harness import validate_baseline
from eval.mmlu import (
    _format_question,
    _label_token_ids,
    build_fewshot_prefix,
    expected_calibration_error,
    score_mmlu,
)


class FakeTokenizer:
    """Minimal tokenizer: distinct ids for " A".." D"."""

    def __init__(self):
        self._vocab = {" A": 10, " B": 11, " C": 12, " D": 13}
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        if text in self._vocab:
            return [self._vocab[text]]
        return [2]

    def __call__(self, text, return_tensors=None):
        n = max(len(text.split()), 1)
        return FakeEncoding({
            "input_ids": torch.zeros(1, n, dtype=torch.long),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
        })


class FakeEncoding(dict):
    """Stands in for transformers' BatchEncoding, which supports .to(device)."""

    def to(self, _device):
        return self


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Returns a fixed answer distribution; used to verify scoring logic."""

    def __init__(self, answer_index, confidence=8.0, vocab=32):
        self.answer_index = answer_index
        self.confidence = confidence
        self.vocab = vocab
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._param

    def __call__(self, **kwargs):
        logits = torch.zeros(1, 4, self.vocab)
        logits[0, -1, 10 + self.answer_index] = self.confidence
        return FakeOutput(logits)


def _question(answer, prefix=""):
    return {
        "subject": "anatomy", "category": "recall",
        "question": "Q?", "choices": ["a", "b", "c", "d"],
        "answer": answer, "prefix": prefix,
    }


class TestPromptFormatting:
    def test_question_ends_with_answer_cue(self):
        assert _format_question("Q?", ["a", "b", "c", "d"]).endswith("Answer:")

    def test_all_four_choices_are_labelled(self):
        text = _format_question("Q?", ["w", "x", "y", "z"])
        for label, choice in zip("ABCD", "wxyz", strict=True):
            assert f"{label}. {choice}" in text

    def test_fewshot_prefix_includes_answers(self):
        dev = [{"question": "Q1", "choices": ["a", "b", "c", "d"], "answer": 2}]
        assert build_fewshot_prefix(dev, n_shot=1).rstrip().endswith("C")

    def test_fewshot_prefix_respects_shot_count(self):
        dev = [{"question": f"Q{i}", "choices": ["a", "b", "c", "d"], "answer": 0}
               for i in range(5)]
        assert build_fewshot_prefix(dev, n_shot=3).count("Answer:") == 3

    def test_empty_dev_gives_empty_prefix(self):
        assert build_fewshot_prefix([], n_shot=5) == ""


class TestLabelTokens:
    def test_labels_map_to_distinct_ids(self):
        assert _label_token_ids(FakeTokenizer()) == [10, 11, 12, 13]

    def test_colliding_labels_raise(self):
        class Collide(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                return [7]

        with pytest.raises(ValueError, match="distinct tokens"):
            _label_token_ids(Collide())


class TestScoreMMLU:
    def test_perfect_model_scores_one(self):
        questions = [_question(2) for _ in range(5)]
        result = score_mmlu(FakeModel(2), FakeTokenizer(), questions, show_progress=False)
        assert result["mmlu_accuracy"] == 1.0

    def test_always_wrong_model_scores_zero(self):
        questions = [_question(0) for _ in range(5)]
        result = score_mmlu(FakeModel(3), FakeTokenizer(), questions, show_progress=False)
        assert result["mmlu_accuracy"] == 0.0

    def test_verbose_response_cannot_break_scoring(self):
        """Regression: the old parser read decoded[0], so 'The answer is B'
        scored as 'T' and was always wrong. Logit scoring never sees text."""
        questions = [_question(1) for _ in range(4)]
        result = score_mmlu(FakeModel(1), FakeTokenizer(), questions, show_progress=False)
        assert result["mmlu_accuracy"] == 1.0

    def test_per_question_flags_support_paired_tests(self):
        questions = [_question(i % 4) for i in range(8)]
        result = score_mmlu(FakeModel(0), FakeTokenizer(), questions, show_progress=False)
        assert len(result["correct_flags"]) == 8
        assert set(result["correct_flags"]) <= {0, 1}

    def test_category_breakdown_is_populated(self):
        questions = [_question(1) for _ in range(3)]
        result = score_mmlu(FakeModel(1), FakeTokenizer(), questions, show_progress=False)
        assert result["by_category"]["recall"] == 1.0


class TestCalibration:
    def test_perfect_calibration_has_low_error(self):
        # Confidence 1.0 paired with 100% accuracy
        assert expected_calibration_error([1.0] * 50, [1] * 50) == pytest.approx(0.0)

    def test_overconfident_wrong_model_has_high_error(self):
        ece = expected_calibration_error([0.99] * 50, [0] * 50)
        assert ece > 0.9

    def test_empty_input_returns_zero(self):
        assert expected_calibration_error([], []) == 0.0


class TestBaselineValidation:
    def test_at_chance_accuracy_fails(self):
        """The exact failure the previous harness shipped: 0.24 accuracy for a
        model published at 0.63, reported as a finding rather than a bug."""
        verdict = validate_baseline("llama_fp16", 0.24, 0.16, 0.26)
        assert verdict["passed"] is False
        assert verdict["at_chance"] is True

    def test_published_level_accuracy_passes(self):
        verdict = validate_baseline("llama_fp16", 0.61, 0.55, 0.67)
        assert verdict["passed"] is True

    def test_far_below_published_fails_with_message(self):
        verdict = validate_baseline("llama_fp16", 0.40, 0.34, 0.46)
        assert verdict["passed"] is False
        assert "published" in verdict["message"]

    def test_unknown_variant_is_not_gated(self):
        verdict = validate_baseline("llama_nf4", 0.55, 0.49, 0.61)
        assert verdict["passed"] is True
