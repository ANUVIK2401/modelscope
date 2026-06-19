import gc
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.configs import MODEL_REGISTRY


def load_model(
    variant_key: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict[str, Any]]:
    if variant_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown variant '{variant_key}'. Valid keys: {list(MODEL_REGISTRY)}"
        )

    model_id, bnb_config = MODEL_REGISTRY[variant_key]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.reset_peak_memory_stats()
    mem_before_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    load_kwargs: dict[str, Any] = {"device_map": "auto"}
    if bnb_config is None:
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["quantization_config"] = bnb_config

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    load_time_s = time.perf_counter() - t0

    mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    metadata: dict[str, Any] = {
        "variant_key": variant_key,
        "model_id": model_id,
        "load_time_s": round(load_time_s, 3),
        "memory_delta_mb": round(mem_peak_mb - mem_before_mb, 1),
    }
    return model, tokenizer, metadata


def unload_model(model: AutoModelForCausalLM) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.generate import generate_text

    print("Loading llama_fp16 ...")
    model, tokenizer, metadata = load_model("llama_fp16")
    print("Metadata:", metadata)

    text, n_tokens, gen_time = generate_text(
        model,
        tokenizer,
        "Explain the attention mechanism in transformers.",
        max_new_tokens=10,
    )
    print(f"Generated ({n_tokens} tokens, {gen_time:.3f}s): {text}")

    unload_model(model)
    print("Model unloaded.")
