"""Model loading with verified memory accounting.

The previous implementation reported 24 MB for a 2B-parameter FP16 model. Two
causes: (a) `mem_before` was sampled after `reset_peak_memory_stats()` and
subtracted from the peak, and (b) `torch.cuda.max_memory_allocated()` only sees
the *caching allocator*, while `device_map="auto"` dispatch can place weights
without every byte passing through it.

This version measures with `torch.cuda.mem_get_info()` (driver-level free VRAM,
which sees everything), cross-checks against the allocator, and asserts the
result is within tolerance of the theoretical weight size.
"""

import gc
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.configs import MODEL_REGISTRY, theoretical_weight_mb

# Measured weight memory must land within this factor of theoretical. 4-bit
# needs the loosest bound: bitsandbytes stores absmax scale constants and keeps
# some layers (embeddings, lm_head) in higher precision, so real usage runs
# meaningfully above the nominal bits/param figure.
_MEMORY_TOLERANCE: dict[str, tuple[float, float]] = {
    "fp16": (0.85, 1.35),
    "int8": (0.85, 1.80),
    "fp4":  (0.85, 2.60),
    "nf4":  (0.85, 2.60),
}


def _device_used_mb() -> float:
    """VRAM in use on device 0, from the driver. Sees all allocations."""
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return (total_bytes - free_bytes) / (1024 ** 2)


def build_load_kwargs(family: str, bnb_config: Any) -> dict[str, Any]:
    """Assemble `from_pretrained` kwargs for a family/quantization pair.

    Split out from the loader so the Gemma attention rule is testable without
    a GPU or a model download.

    Gemma-2 soft-caps attention logits. The SDPA and flash-attention kernels
    silently drop that capping, and transformers picks SDPA by default when it
    is available -- so every Gemma quality number would describe an
    architecture the model was never trained as. Only the eager path
    implements it. Llama has no soft-capping and keeps the faster kernel.
    """
    kwargs: dict[str, Any] = {"device_map": {"": 0}}
    if family == "gemma":
        kwargs["attn_implementation"] = "eager"
    if bnb_config is None:
        # `dtype` is the current name; `torch_dtype` is back-compat only.
        kwargs["dtype"] = torch.float16
    else:
        kwargs["quantization_config"] = bnb_config
    return kwargs


def measure_load_memory(variant_key: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load a variant and measure the VRAM its weights actually occupy.

    Returns (model, tokenizer, metadata). Metadata carries both the driver-level
    and allocator-level readings; they should agree closely, and a large gap is
    itself a signal worth recording.
    """
    if variant_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown variant '{variant_key}'. Valid keys: {list(MODEL_REGISTRY)}"
        )

    model_id, family, config_name, bnb_config = MODEL_REGISTRY[variant_key]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models must left-pad, or batched generation decodes from
    # pad tokens and produces garbage. This silently corrupted every batched
    # throughput measurement in the previous version.
    tokenizer.padding_side = "left"

    # Settle the allocator so the baseline is clean.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    driver_before_mb = _device_used_mb()
    alloc_before_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    load_kwargs = build_load_kwargs(family, bnb_config)

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    torch.cuda.synchronize()
    load_time_s = time.perf_counter() - t0

    driver_after_mb = _device_used_mb()
    alloc_after_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    weight_mb_driver = driver_after_mb - driver_before_mb
    weight_mb_alloc = alloc_after_mb - alloc_before_mb

    # Driver reading is authoritative: it captures allocations the caching
    # allocator never sees.
    weight_mb = weight_mb_driver

    theoretical_mb = theoretical_weight_mb(variant_key)
    ratio = weight_mb / theoretical_mb if theoretical_mb > 0 else 0.0
    lo, hi = _MEMORY_TOLERANCE[config_name]

    metadata: dict[str, Any] = {
        "variant": variant_key,
        "model_id": model_id,
        "model": family,
        "config": config_name,
        "load_time_s": round(load_time_s, 3),
        "weight_memory_mb": round(weight_mb, 1),
        "weight_memory_mb_allocator": round(weight_mb_alloc, 1),
        "theoretical_weight_mb": round(theoretical_mb, 1),
        "memory_ratio_vs_theoretical": round(ratio, 3),
        "memory_check_passed": bool(lo <= ratio <= hi),
    }

    if not metadata["memory_check_passed"]:
        raise RuntimeError(
            f"Memory measurement failed sanity check for {variant_key}: "
            f"measured {weight_mb:.1f} MB vs theoretical {theoretical_mb:.1f} MB "
            f"(ratio {ratio:.2f}, expected {lo}-{hi}). "
            "Refusing to record a physically implausible number."
        )

    return model, tokenizer, metadata


def load_model(variant_key: str) -> tuple[Any, Any, dict[str, Any]]:
    """Convenience wrapper for callers that only need the model."""
    return measure_load_memory(variant_key)


def unload_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def measure_kv_cache_mb(
    model: Any,
    tokenizer: Any,
    batch_size: int,
    seq_len: int,
) -> float:
    """VRAM the KV cache occupies at a given batch and sequence length.

    Weight memory alone understates deployment footprint. At batch 32 the KV
    cache can exceed the weights, which inverts the memory ranking between
    configs -- 4-bit shrinks weights but leaves the FP16 cache untouched.
    """
    prompt_ids = torch.randint(
        0, tokenizer.vocab_size, (batch_size, seq_len), device=model.device
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before_mb = _device_used_mb()

    with torch.no_grad():
        out = model(prompt_ids, use_cache=True)

    torch.cuda.synchronize()
    after_mb = _device_used_mb()

    del out
    gc.collect()
    torch.cuda.empty_cache()
    return round(after_mb - before_mb, 1)
