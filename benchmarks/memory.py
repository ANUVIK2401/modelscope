from typing import Any

import torch

from models.loader import load_model, unload_model


def measure_memory(variant_key: str) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    model, _tokenizer, _metadata = load_model(variant_key)
    memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    unload_model(model)
    return {
        "variant_key": variant_key,
        "memory_mb": round(memory_mb, 1),
    }
