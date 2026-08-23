"""MMLU scoring via answer-token log-likelihood (standard protocol).

The previous implementation generated free text and read `decoded[0]`, so
"The answer is B" scored as "T" -> wrong. Combined with 0-shot prompting it
produced 0.13-0.37 accuracy, at or below the 0.25 random-guess floor, for models
that score ~0.60 in published evals. Those numbers measured the parser, not the
model.

This implements the protocol used by the HELM / lm-evaluation-harness family and
by the model cards we compare against:
  - 5-shot prompt built from the dataset's own `dev` split (exactly 5 rows/subject)
  - score = argmax over the logits of the " A"/" B"/" C"/" D" continuation tokens
  - no generation, no parsing, no chat template

Scoring the four candidate tokens directly makes the metric independent of
instruction-following ability, which is what lets a quantized base model be
compared against its FP16 self without confounding the two effects.
"""

import random
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from models.configs import EVAL_SUBJECTS
from models.env import SEED

_LABELS = ("A", "B", "C", "D")
_N_SHOT = 5
_QUESTIONS_PER_SUBJECT = 40


def _format_question(question: str, choices: list[str]) -> str:
    lines = [question.strip()]
    for label, choice in zip(_LABELS, choices, strict=False):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def _subject_header(subject: str) -> str:
    pretty = subject.replace("_", " ")
    return f"The following are multiple choice questions (with answers) about {pretty}.\n\n"


def build_fewshot_prefix(dev_rows: list[dict[str, Any]], n_shot: int = _N_SHOT) -> str:
    """Few-shot prefix from the dev split, with answers filled in."""
    blocks = []
    for row in dev_rows[:n_shot]:
        block = _format_question(row["question"], row["choices"])
        blocks.append(f"{block} {_LABELS[int(row['answer'])]}")
    return "\n\n".join(blocks) + "\n\n" if blocks else ""


def load_mmlu_questions(
    questions_per_subject: int = _QUESTIONS_PER_SUBJECT,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Load MMLU test questions plus each subject's 5-shot prefix.

    Sampling is seeded and recorded so the same question set is reused across
    every variant -- a paired design, which is what makes the paired
    significance tests in analysis/statistics.py valid.
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []

    for subject, category in EVAL_SUBJECTS.items():
        test_ds = load_dataset("cais/mmlu", subject, split="test")
        dev_ds = load_dataset("cais/mmlu", subject, split="dev")
        dev_rows = [dict(dev_ds[i]) for i in range(len(dev_ds))]
        prefix = _subject_header(subject) + build_fewshot_prefix(dev_rows)

        n = min(questions_per_subject, len(test_ds))
        indices = sorted(rng.sample(range(len(test_ds)), n))
        for i in indices:
            row = test_ds[i]
            questions.append({
                "subject": subject,
                "category": category,
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),
                "prefix": prefix,
            })

    return questions


def _label_token_ids(tokenizer: Any) -> list[int]:
    """Token ids for the answer continuations.

    The leading space matters: after "Answer:" the model emits " A", which most
    BPE tokenizers encode as a single token distinct from "A". Getting this
    wrong silently scores the wrong distribution.
    """
    ids = []
    for label in _LABELS:
        candidates = tokenizer.encode(f" {label}", add_special_tokens=False)
        if not candidates:
            candidates = tokenizer.encode(label, add_special_tokens=False)
        # Take the last token: some tokenizers prepend a metaspace marker.
        ids.append(candidates[-1])
    if len(set(ids)) != len(_LABELS):
        raise ValueError(
            f"Answer labels do not map to distinct tokens: {ids}. "
            "Scoring would be ambiguous."
        )
    return ids


@torch.no_grad()
def score_mmlu(
    model: Any,
    tokenizer: Any,
    questions: list[dict[str, Any]],
    show_progress: bool = True,
) -> dict[str, Any]:
    """Score MMLU by answer-token log-likelihood.

    Returns overall accuracy, per-subject and per-category accuracy, and the
    per-question correctness vector. That vector is the input to the paired
    bootstrap -- keeping it is what allows "is int4 worse than fp16" to be
    answered with a confidence interval instead of a point estimate.
    """
    device = next(model.parameters()).device
    label_ids = _label_token_ids(tokenizer)

    correct_flags: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []

    iterator = tqdm(questions, desc="MMLU", leave=False) if show_progress else questions
    for item in iterator:
        prompt = item["prefix"] + _format_question(item["question"], item["choices"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        logits = model(**inputs).logits[0, -1, :]
        answer_logits = torch.tensor(
            [logits[i].item() for i in label_ids], dtype=torch.float32
        )
        probs = torch.softmax(answer_logits, dim=-1)

        predicted = int(torch.argmax(answer_logits).item())
        predictions.append(predicted)
        confidences.append(float(probs[predicted].item()))
        correct_flags.append(int(predicted == item["answer"]))

    flags = np.array(correct_flags, dtype=float)
    accuracy = float(flags.mean()) if len(flags) else 0.0

    by_subject: dict[str, float] = {}
    by_category: dict[str, float] = {}
    for key, field in (("subject", by_subject), ("category", by_category)):
        groups: dict[str, list[int]] = {}
        for item, flag in zip(questions, correct_flags, strict=True):
            groups.setdefault(item[key], []).append(flag)
        for name, values in groups.items():
            field[name] = round(float(np.mean(values)), 4)

    return {
        "mmlu_accuracy": round(accuracy, 4),
        "n_questions": len(questions),
        "by_subject": by_subject,
        "by_category": by_category,
        "correct_flags": correct_flags,
        "predictions": predictions,
        "confidences": confidences,
        "answers": [q["answer"] for q in questions],
    }


def expected_calibration_error(
    confidences: list[float],
    correct_flags: list[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error over confidence bins.

    Quantization can preserve accuracy while distorting the output distribution.
    A model that is still right but no longer knows when it is right is a
    deployment hazard that accuracy alone does not surface.
    """
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct_flags, dtype=float)
    # Confidence over 4 classes is bounded below by 0.25.
    edges = np.linspace(0.25, 1.0, n_bins + 1)
    total = len(conf)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        weight = mask.sum() / total
        ece += weight * abs(acc[mask].mean() - conf[mask].mean())
    return round(float(ece), 4)
