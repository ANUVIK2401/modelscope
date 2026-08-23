"""WikiText-2 perplexity via strided sliding window.

The previous implementation scored each document independently with
`max_length=512, truncation=True`, discarding everything past 512 tokens and
giving every document equal weight regardless of length. That is not the
protocol the GPTQ/AWQ papers use, so the numbers were not comparable to the
published results the README claimed comparability with.

Standard protocol: concatenate the corpus, slide a fixed context window with
stride, and score only the non-overlapping tokens in each window so every token
is predicted exactly once with maximal available context.
"""

import math
from typing import Any

import torch
from tqdm import tqdm

_MAX_LENGTH = 2048
_STRIDE = 1024


@torch.no_grad()
def measure_perplexity(
    model: Any,
    tokenizer: Any,
    max_length: int = _MAX_LENGTH,
    stride: int = _STRIDE,
    max_tokens: int | None = 200_000,
) -> dict[str, float]:
    """Sliding-window perplexity on WikiText-2 test.

    `max_tokens` caps the corpus for tractability on a T4; the cap is recorded
    in the result so the number is reproducible rather than silently truncated.
    """
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    corpus = "\n\n".join(dataset["text"])
    encodings = tokenizer(corpus, return_tensors="pt")
    input_ids_full = encodings.input_ids
    if max_tokens is not None:
        input_ids_full = input_ids_full[:, :max_tokens]

    seq_len = int(input_ids_full.shape[1])
    device = next(model.parameters()).device

    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    for begin in tqdm(range(0, seq_len, stride), desc="perplexity", leave=False):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # tokens scored in this window
        if target_len <= 0:
            continue

        input_ids = input_ids_full[:, begin:end].to(device)
        target_ids = input_ids.clone()
        # Mask everything already scored by a previous window.
        target_ids[:, :-target_len] = -100

        out = model(input_ids, labels=target_ids)

        # HF averages over valid label positions; one is lost to the shift.
        n_valid = max(target_len - 1, 1)
        nll_sum += float(out.loss.item()) * n_valid
        n_tokens += n_valid

        prev_end = end
        if end == seq_len:
            break

    if n_tokens == 0:
        raise RuntimeError("Perplexity scored zero tokens - corpus or tokenizer issue")

    mean_nll = nll_sum / n_tokens
    return {
        "perplexity": round(math.exp(mean_nll), 4),
        "perplexity_nll": round(mean_nll, 6),
        "perplexity_tokens": n_tokens,
        "perplexity_window": max_length,
        "perplexity_stride": stride,
    }
