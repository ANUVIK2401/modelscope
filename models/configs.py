import torch
from transformers import BitsAndBytesConfig

_LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
_GEMMA = "google/gemma-3-4b-it"

MODEL_REGISTRY: dict[str, tuple[str, BitsAndBytesConfig | None]] = {
    "llama_fp16":     (_LLAMA, None),
    "llama_int8":     (_LLAMA, BitsAndBytesConfig(load_in_8bit=True)),
    "llama_int4":     (_LLAMA, BitsAndBytesConfig(
                          load_in_4bit=True,
                          bnb_4bit_compute_dtype=torch.float16,
                      )),
    "llama_int4_nf4": (_LLAMA, BitsAndBytesConfig(
                          load_in_4bit=True,
                          bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.float16,
                      )),
    "gemma_fp16":     (_GEMMA, None),
    "gemma_int8":     (_GEMMA, BitsAndBytesConfig(load_in_8bit=True)),
    "gemma_int4":     (_GEMMA, BitsAndBytesConfig(
                          load_in_4bit=True,
                          bnb_4bit_compute_dtype=torch.float16,
                      )),
    "gemma_int4_nf4": (_GEMMA, BitsAndBytesConfig(
                          load_in_4bit=True,
                          bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.float16,
                      )),
}

BENCHMARK_PROMPTS: list[str] = [
    "Explain the attention mechanism in transformers.",
    "What are the trade-offs between gradient descent optimizers like Adam and SGD?",
    "Write a Python function that merges two sorted lists in O(n) time.",
]

BATCH_SIZES: list[int] = [1, 4, 16, 32]

EVAL_SUBJECTS: list[str] = [
    "high_school_mathematics",
    "world_history",
    "computer_science",
    "high_school_biology",
]
