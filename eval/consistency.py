"""Output consistency under stochastic decoding.

The previous code defined this metric two incompatible ways under one column
name: eval/consistency.py returned `1 - std(rouge)` with greedy decoding (which
makes all outputs identical, std=0, so the metric was constant 1.0 -- degenerate
and measuring nothing), while eval/harness.py returned `mean(rouge)` with
temperature sampling. Different quantity, same CSV column.

Definition used here: mean pairwise ROUGE-L F1 across N sampled generations at
fixed temperature. Higher means the model's output distribution is tighter.
Greedy decoding is explicitly rejected -- it is deterministic, so there is no
variance to measure.
"""

import itertools
from typing import Any

import numpy as np
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

_MAX_NEW_TOKENS = 100
_TEMPERATURE = 0.7
_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


@torch.no_grad()
def measure_consistency(
    model: Any,
    tokenizer: Any,
    prompt: str,
    runs: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Mean pairwise ROUGE-L across `runs` sampled generations.

    Each run is seeded distinctly-but-reproducibly so the sample is stochastic
    within a variant yet identical across variants -- the comparison stays
    controlled.
    """
    if runs < 2:
        raise ValueError("Consistency needs at least 2 generations to compare")

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = int(inputs["input_ids"].shape[1])

    outputs: list[str] = []
    for i in tqdm(range(runs), desc="consistency", leave=False):
        torch.manual_seed(seed + i)
        ids = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=_TEMPERATURE,
            top_p=0.95,
            pad_token_id=pad_id,
        )
        outputs.append(tokenizer.decode(ids[0][input_len:], skip_special_tokens=True))

    pairwise = [
        _SCORER.score(a, b)["rougeL"].fmeasure
        for a, b in itertools.combinations(outputs, 2)
    ]
    scores = np.asarray(pairwise, dtype=float)

    return {
        "consistency_score": round(float(scores.mean()), 4),
        "consistency_std": round(float(scores.std(ddof=1)) if scores.size > 1 else 0.0, 4),
        "consistency_n_pairs": int(scores.size),
    }
