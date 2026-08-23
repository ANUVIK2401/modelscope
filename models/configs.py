"""Model and quantization variant registry.

Naming note: bitsandbytes' `bnb_4bit_quant_type` defaults to "fp4", NOT "nf4".
The previous version of this file named the default-4bit variant `int4` and the
explicit one `int4_nf4`, which read as "4-bit" vs "4-bit with the good format".
They are actually two distinct 4-bit *formats* -- fp4 and nf4 -- and the
comparison between them is one of this study's research questions. Variants are
now named for the format they actually use.
"""

import torch
from transformers import BitsAndBytesConfig

LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
GEMMA = "google/gemma-2-2b-it"

# Parameter counts used for the memory sanity check. Sourced from the model
# cards; the assertion tolerance is wide enough to absorb small discrepancies.
PARAM_COUNTS: dict[str, float] = {
    LLAMA: 3.21e9,
    GEMMA: 2.61e9,
}

# Bits per stored weight for each quantization config. bitsandbytes keeps a
# small fraction of parameters in higher precision (LLM.int8() outlier channels,
# and the nested absmax constants for 4-bit), so these are nominal.
NOMINAL_BITS: dict[str, float] = {
    "fp16": 16.0,
    "int8": 8.0,
    "fp4": 4.0,
    "nf4": 4.0,
}


def _bnb_4bit(quant_type: str, double_quant: bool = False) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=double_quant,
    )


# variant_key -> (model_id, model_family, config_name, quantization_config)
MODEL_REGISTRY: dict[str, tuple[str, str, str, BitsAndBytesConfig | None]] = {
    "llama_fp16": (LLAMA, "llama", "fp16", None),
    "llama_int8": (LLAMA, "llama", "int8", BitsAndBytesConfig(load_in_8bit=True)),
    "llama_fp4":  (LLAMA, "llama", "fp4",  _bnb_4bit("fp4")),
    "llama_nf4":  (LLAMA, "llama", "nf4",  _bnb_4bit("nf4")),
    "gemma_fp16": (GEMMA, "gemma", "fp16", None),
    "gemma_int8": (GEMMA, "gemma", "int8", BitsAndBytesConfig(load_in_8bit=True)),
    "gemma_fp4":  (GEMMA, "gemma", "fp4",  _bnb_4bit("fp4")),
    "gemma_nf4":  (GEMMA, "gemma", "nf4",  _bnb_4bit("nf4")),
}

ALL_VARIANTS: list[str] = list(MODEL_REGISTRY)

# The FP16 run of each family is the reference point every degradation metric
# is measured against.
BASELINE_VARIANT: dict[str, str] = {
    "llama": "llama_fp16",
    "gemma": "gemma_fp16",
}


def theoretical_weight_mb(variant_key: str) -> float:
    """Expected weight memory in MB, from parameter count and bit width.

    The measured peak must land near this. A 2B model reading 24 MB is not a
    finding about the model -- it is a broken measurement, and the runner
    asserts against this value to make that failure loud.
    """
    model_id, _family, config_name, _bnb = MODEL_REGISTRY[variant_key]
    params = PARAM_COUNTS[model_id]
    bits = NOMINAL_BITS[config_name]
    return (params * bits / 8) / (1024 ** 2)


# Prompts for latency/throughput. All three are used; averaging over prompts
# keeps a single prompt's tokenization from driving the headline number.
BENCHMARK_PROMPTS: list[str] = [
    "Explain the attention mechanism in transformers.",
    "What are the trade-offs between gradient descent optimizers like Adam and SGD?",
    "Write a Python function that merges two sorted lists in O(n) time.",
]

BATCH_SIZES: list[int] = [1, 4, 16, 32]

# MMLU subjects, chosen to span reasoning-heavy and recall-heavy tasks so
# task-conditional degradation is measurable rather than averaged away.
# `category` drives the per-capability breakdown in the analysis.
EVAL_SUBJECTS: dict[str, str] = {
    "high_school_mathematics": "reasoning",
    "abstract_algebra":        "reasoning",
    "formal_logic":            "reasoning",
    "world_religions":         "recall",
    "high_school_geography":   "recall",
    "anatomy":                 "recall",
    "computer_security":       "applied",
    "high_school_biology":     "applied",
}

SUBJECT_CATEGORIES: list[str] = ["reasoning", "recall", "applied"]
