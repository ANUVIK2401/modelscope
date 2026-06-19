import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
) -> tuple[str, int, float]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.perf_counter() - t0

    new_ids = output_ids[0][input_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, len(new_ids), gen_time
