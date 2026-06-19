import itertools

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

_MAX_NEW_TOKENS = 100
_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def measure_consistency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    runs: int = 5,
) -> float:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    outputs: list[str] = []
    for _ in range(runs):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][input_len:]
        outputs.append(tokenizer.decode(new_ids, skip_special_tokens=True))

    # Compute ROUGE-L F1 for every unordered pair of outputs.
    rouge_scores = [
        _SCORER.score(a, b)["rougeL"].fmeasure
        for a, b in itertools.combinations(outputs, 2)
    ]

    # 1 - std: zero variance → score = 1.0 (perfectly consistent);
    # high variance → score approaches 0.5 (max achievable std for [0,1] values).
    return float(1.0 - np.std(rouge_scores))
