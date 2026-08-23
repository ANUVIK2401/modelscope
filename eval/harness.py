"""Quality evaluation orchestrator.

Runs MMLU (logit-scored, 5-shot), consistency, and perplexity for each variant,
and validates every FP16 baseline against the published MMLU figure before any
downstream analysis is allowed to use the results.

That gate is the check that was missing: the previous harness produced 0.13-0.37
accuracy -- at or below the 0.25 random-guess floor -- and nothing in the
pipeline objected, so broken numbers reached the README as findings.
"""

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from analysis.paths import QUALITY_CSV, RESULTS_DIR, ensure_results_dir
from analysis.statistics import accuracy_ci
from benchmarks.runner import append_row, completed_variants
from eval.consistency import measure_consistency
from eval.mmlu import expected_calibration_error, load_mmlu_questions, score_mmlu
from eval.perplexity import measure_perplexity
from models.configs import ALL_VARIANTS, MODEL_REGISTRY
from models.env import capture_environment, environment_hash, set_global_seed
from models.loader import measure_load_memory, unload_model

CONSISTENCY_PROMPT = (
    "What are the key differences between supervised and unsupervised learning?"
)

# Published 5-shot MMLU from the model cards. An FP16 run landing far below
# these means the harness is broken, not the model.
PUBLISHED_MMLU: dict[str, float] = {
    "llama_fp16": 0.63,   # Llama-3.2-3B-Instruct
    "gemma_fp16": 0.57,   # Gemma-2-2b-it
}
_BASELINE_TOLERANCE = 0.12
_CHANCE_LEVEL = 0.25

# Per-question correctness vectors, needed for the paired significance tests.
PER_QUESTION_JSON = RESULTS_DIR / "per_question_flags.json"


def validate_baseline(variant_key: str, accuracy: float,
                      ci_low: float, ci_high: float) -> dict[str, Any]:
    """Check an FP16 run against its published number.

    Returns a verdict dict; the caller decides whether to abort. Two separate
    failures are distinguished: at-chance (parser/protocol broken) and merely
    off-target (possible but needs explanation).
    """
    verdict: dict[str, Any] = {
        "variant": variant_key,
        "measured": round(accuracy, 4),
        "ci": [round(ci_low, 4), round(ci_high, 4)],
        "published": PUBLISHED_MMLU.get(variant_key),
        "at_chance": bool(ci_high <= _CHANCE_LEVEL + 0.02),
        "passed": True,
        "message": "",
    }

    if verdict["at_chance"]:
        verdict["passed"] = False
        verdict["message"] = (
            f"{variant_key} scores {accuracy:.3f}, indistinguishable from the "
            f"{_CHANCE_LEVEL} random-guess floor. The eval protocol is broken."
        )
        return verdict

    published = verdict["published"]
    if published is not None and abs(accuracy - published) > _BASELINE_TOLERANCE:
        verdict["passed"] = False
        verdict["message"] = (
            f"{variant_key} scores {accuracy:.3f} vs published {published:.3f} "
            f"(tolerance {_BASELINE_TOLERANCE}). Investigate before trusting "
            "downstream comparisons."
        )
    return verdict


def evaluate_variant(variant_key: str, questions: list[dict[str, Any]],
                     env_hash: str, skip_perplexity: bool = False) -> tuple[dict[str, Any], list[int]]:
    _model_id, family, config_name, _bnb = MODEL_REGISTRY[variant_key]

    model, tokenizer, meta = measure_load_memory(variant_key)
    try:
        mmlu = score_mmlu(model, tokenizer, questions)
        consistency = measure_consistency(model, tokenizer, CONSISTENCY_PROMPT)
        perplexity = (
            {"perplexity": None} if skip_perplexity
            else measure_perplexity(model, tokenizer)
        )
    finally:
        unload_model(model)

    acc, lo, hi = accuracy_ci(mmlu["correct_flags"])
    ece = expected_calibration_error(mmlu["confidences"], mmlu["correct_flags"])

    row: dict[str, Any] = {
        "variant": variant_key,
        "model": family,
        "config": config_name,
        "status": "ok",
        "error": "",
        "env_hash": env_hash,
        "mmlu_accuracy": round(acc, 4),
        "mmlu_ci_low": round(lo, 4),
        "mmlu_ci_high": round(hi, 4),
        "mmlu_n": mmlu["n_questions"],
        "ece": ece,
        **{f"mmlu_{cat}": val for cat, val in mmlu["by_category"].items()},
        **consistency,
        **perplexity,
        "load_time_s": meta["load_time_s"],
    }
    return row, mmlu["correct_flags"]


def run(variants: list[str], csv_path: Path = QUALITY_CSV,
        resume: bool = True, skip_perplexity: bool = False,
        strict: bool = True) -> None:
    set_global_seed()
    ensure_results_dir()

    env_hash = environment_hash(capture_environment())
    questions = load_mmlu_questions()
    print(f"Loaded {len(questions)} MMLU questions across "
          f"{len({q['subject'] for q in questions})} subjects (5-shot, logit-scored)")

    done = completed_variants(csv_path) if resume else set()
    flags_store: dict[str, list[int]] = {}
    if PER_QUESTION_JSON.exists():
        flags_store = json.loads(PER_QUESTION_JSON.read_text())

    verdicts: list[dict[str, Any]] = []

    for variant_key in tqdm(variants, desc="variants"):
        if variant_key in done:
            continue
        try:
            row, flags = evaluate_variant(
                variant_key, questions, env_hash, skip_perplexity
            )
            append_row(csv_path, row)
            flags_store[variant_key] = flags
            PER_QUESTION_JSON.write_text(json.dumps(flags_store))

            print(f"{variant_key}: MMLU {row['mmlu_accuracy']:.3f} "
                  f"[{row['mmlu_ci_low']:.3f}, {row['mmlu_ci_high']:.3f}], "
                  f"ECE {row['ece']:.3f}, PPL {row['perplexity']}")

            if variant_key in PUBLISHED_MMLU:
                verdict = validate_baseline(
                    variant_key, row["mmlu_accuracy"],
                    row["mmlu_ci_low"], row["mmlu_ci_high"],
                )
                verdicts.append(verdict)
                if not verdict["passed"]:
                    print(f"BASELINE CHECK FAILED: {verdict['message']}")
                    if strict:
                        raise RuntimeError(verdict["message"])
                else:
                    print(f"baseline check passed for {variant_key}")

        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            if strict and isinstance(exc, RuntimeError) and "baseline" in str(exc).lower():
                raise
            traceback.print_exc()
            _model_id, family, config_name, _bnb = MODEL_REGISTRY[variant_key]
            append_row(csv_path, {
                "variant": variant_key, "model": family, "config": config_name,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:500],
                "env_hash": env_hash,
            })
            print(f"FAILED {variant_key}: {type(exc).__name__}: {exc}")

    if verdicts:
        (RESULTS_DIR / "baseline_validation.json").write_text(
            json.dumps(verdicts, indent=2)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality evaluation.")
    parser.add_argument("--variants", nargs="*", default=ALL_VARIANTS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-perplexity", action="store_true",
                        help="Skip WikiText-2 perplexity (the slowest metric).")
    parser.add_argument("--no-strict", action="store_true",
                        help="Continue even if an FP16 baseline fails validation.")
    parser.add_argument("--output", type=Path, default=QUALITY_CSV)
    args = parser.parse_args()
    run(args.variants, csv_path=args.output, resume=not args.no_resume,
        skip_perplexity=args.skip_perplexity, strict=not args.no_strict)


if __name__ == "__main__":
    main()
