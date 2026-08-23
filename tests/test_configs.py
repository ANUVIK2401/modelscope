import pytest

from models.configs import (
    ALL_VARIANTS,
    EVAL_SUBJECTS,
    GEMMA,
    LLAMA,
    MODEL_REGISTRY,
    SUBJECT_CATEGORIES,
    theoretical_weight_mb,
)


class TestVariantNaming:
    def test_fp4_and_nf4_are_distinct_configs(self):
        """bitsandbytes defaults bnb_4bit_quant_type to 'fp4'. The old naming
        ('int4' vs 'int4_nf4') hid that these are two different 4-bit formats,
        which made the comparison between them unreadable."""
        _id, _fam, _cfg, fp4 = MODEL_REGISTRY["llama_fp4"]
        _id, _fam, _cfg, nf4 = MODEL_REGISTRY["llama_nf4"]
        assert fp4.bnb_4bit_quant_type == "fp4"
        assert nf4.bnb_4bit_quant_type == "nf4"
        assert fp4.bnb_4bit_quant_type != nf4.bnb_4bit_quant_type

    def test_no_variant_is_named_ambiguously(self):
        # "int4" alone does not say which 4-bit format is meant
        assert not any(v.endswith("_int4") for v in ALL_VARIANTS)

    def test_every_variant_declares_family_and_config(self):
        for key, (model_id, family, config, _bnb) in MODEL_REGISTRY.items():
            assert key.startswith(family)
            assert config in {"fp16", "int8", "fp4", "nf4"}
            assert model_id in {LLAMA, GEMMA}

    def test_both_families_cover_all_four_configs(self):
        for family in ("llama", "gemma"):
            configs = {c for _i, f, c, _b in MODEL_REGISTRY.values() if f == family}
            assert configs == {"fp16", "int8", "fp4", "nf4"}


class TestTheoreticalMemory:
    def test_fp16_3b_model_is_several_gigabytes(self):
        """Regression: the previous runner reported 24 MB for a 2B FP16 model.
        The theoretical value is what makes that reading assertably wrong."""
        assert theoretical_weight_mb("llama_fp16") > 5000

    def test_gemma_fp16_is_not_tiny(self):
        assert theoretical_weight_mb("gemma_fp16") > 4000

    def test_four_bit_is_about_a_quarter_of_fp16(self):
        ratio = theoretical_weight_mb("llama_fp4") / theoretical_weight_mb("llama_fp16")
        assert ratio == pytest.approx(0.25, abs=0.01)

    def test_int8_is_about_half_of_fp16(self):
        ratio = theoretical_weight_mb("llama_int8") / theoretical_weight_mb("llama_fp16")
        assert ratio == pytest.approx(0.5, abs=0.01)

    def test_fp4_and_nf4_have_identical_nominal_size(self):
        assert theoretical_weight_mb("llama_fp4") == theoretical_weight_mb("llama_nf4")


class TestEvalSubjects:
    def test_subjects_span_all_categories(self):
        assert set(EVAL_SUBJECTS.values()) == set(SUBJECT_CATEGORIES)

    def test_reasoning_and_recall_both_represented(self):
        """Aggregate MMLU averages away task-conditional degradation; the
        split is what makes 'quantization hurts reasoning more' testable."""
        counts = {c: list(EVAL_SUBJECTS.values()).count(c) for c in SUBJECT_CATEGORIES}
        assert counts["reasoning"] >= 2
        assert counts["recall"] >= 2
