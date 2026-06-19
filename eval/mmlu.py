import random
import re
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.configs import EVAL_SUBJECTS

_SEED = 42
_QUESTIONS_PER_SUBJECT = 25
_MAX_NEW_TOKENS = 5
_LABELS = ["A", "B", "C", "D"]
_LABEL_TO_IDX = {label: i for i, label in enumerate(_LABELS)}


def load_mmlu_questions() -> list[dict[str, Any]]:
    rng = random.Random(_SEED)
    questions: list[dict[str, Any]] = []
    for subject in EVAL_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        indices = rng.sample(range(len(ds)), min(_QUESTIONS_PER_SUBJECT, len(ds)))
        for i in indices:
            row = ds[i]
            questions.append({
                "question": row["question"],
                "choices": row["choices"],
                "answer": int(row["answer"]),
                "subject": subject,
            })
    return questions


def _format_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{l}. {c}" for l, c in zip(_LABELS, choices))
    return (
        "Answer the following multiple choice question with only the letter A, B, C, or D.\n\n"
        f"Question: {question}\n{options}\n\nAnswer:"
    )


def _extract_letter(text: str) -> int | None:
    # Take the first A/B/C/D character that appears in the response.
    match = re.search(r"\b([ABCD])\b", text.strip())
    if match:
        return _LABEL_TO_IDX[match.group(1)]
    # Fallback: first occurrence of any label character.
    for ch in text.strip():
        if ch in _LABEL_TO_IDX:
            return _LABEL_TO_IDX[ch]
    return None


def score_mmlu(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: list[dict[str, Any]],
) -> float:
    device = next(model.parameters()).device
    correct = 0

    for item in tqdm(questions, desc="MMLU", leave=False):
        prompt = _format_prompt(item["question"], item["choices"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][input_len:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        predicted = _extract_letter(response)
        if predicted is not None and predicted == item["answer"]:
            correct += 1

    return correct / len(questions) if questions else 0.0
