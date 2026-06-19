import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from eval.consistency import measure_consistency
from eval.mmlu import load_mmlu_questions, score_mmlu
from eval.relevance import measure_relevance
from models.configs import BENCHMARK_PROMPTS, MODEL_REGISTRY
from models.loader import load_model, unload_model

_CSV_PATH = Path("results/quality_results.csv")
_CSV_COLUMNS = [
    "variant",
    "model",
    "config",
    "mmlu_accuracy",
    "consistency_score",
    "relevance_score",
]

_CONSISTENCY_PROMPT = BENCHMARK_PROMPTS[0]
_RELEVANCE_PROMPT = BENCHMARK_PROMPTS[0]
# Concise factual reference used as the BERTScore target for the standard prompt.
_RELEVANCE_REFERENCE = (
    "The attention mechanism computes query, key, and value projections for each token. "
    "Scaled dot-product attention scores between queries and keys are softmax-normalized "
    "to produce weights that blend the value vectors, allowing each position to "
    "selectively attend to other positions in the sequence."
)


def _append_row(row: dict[str, Any]) -> None:
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _CSV_PATH.exists()
    with _CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_full_eval() -> None:
    questions = load_mmlu_questions()
    tqdm.write(f"Loaded {len(questions)} MMLU questions across {len(set(q['subject'] for q in questions))} subjects")

    for variant_key in tqdm(MODEL_REGISTRY, desc="eval", unit="variant"):
        model_name, config_name = variant_key.split("_", 1)

        model, tokenizer, _metadata = load_model(variant_key)

        mmlu_accuracy = score_mmlu(model, tokenizer, questions)
        consistency = measure_consistency(model, tokenizer, _CONSISTENCY_PROMPT)
        relevance = measure_relevance(model, tokenizer, _RELEVANCE_PROMPT, _RELEVANCE_REFERENCE)

        unload_model(model)

        row: dict[str, Any] = {
            "variant": variant_key,
            "model": model_name,
            "config": config_name,
            "mmlu_accuracy": round(mmlu_accuracy, 4),
            "consistency_score": round(consistency, 4),
            "relevance_score": round(relevance, 4),
        }
        _append_row(row)

        tqdm.write(
            f"  {variant_key}: mmlu={row['mmlu_accuracy']:.3f} "
            f"consistency={row['consistency_score']:.3f} "
            f"relevance={row['relevance_score']:.3f}"
        )
