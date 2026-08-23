"""Layer-wise quantization sensitivity: measurement and prediction.

This is the research contribution. The benchmark half of this project answers
"which config should I deploy" -- a useful engineering report, but the answer
does not transfer to a model nobody has benchmarked yet.

The question here transfers: **can the damage quantization does to a layer be
predicted from that layer's weight distribution, without running an eval?**

Mechanism. Round-to-nearest quantization error scales with the dynamic range of
the block being quantized. bitsandbytes quantizes in blocks of 64 with a shared
absmax scale, so a single large-magnitude outlier inflates the scale for its
whole block and coarsens the effective resolution for the other 63 weights. The
prediction is therefore that heavy-tailed weight distributions -- high kurtosis,
large max/median magnitude ratio -- suffer disproportionate quantization error.
That is a falsifiable claim, and this module tests it.

Design:
  1. `layer_weight_statistics` -- cheap statistics from FP16 weights alone.
     No GPU inference, no eval data, seconds to compute.
  2. `layer_quantization_error` -- ground truth: quantize each layer, measure
     the reconstruction error against FP16.
  3. `fit_sensitivity_predictor` -- regress (2) on (1), report held-out R^2.

If the predictor generalizes across model families, cheap weight statistics
substitute for expensive evaluation when triaging which layers to protect.
"""

from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

# bitsandbytes 4-bit block size. Block-level scaling is the mechanism the
# outlier hypothesis operates through, so statistics are computed per block.
BLOCK_SIZE = 64


def _to_float32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(torch.float32).flatten()


def layer_weight_statistics(weight: torch.Tensor) -> dict[str, float]:
    """Distribution statistics for one weight matrix.

    Everything here is computable from the FP16 checkpoint alone -- that is the
    point. If these predict degradation, no eval run is needed to triage layers.
    """
    flat = _to_float32(weight)
    n = flat.numel()
    absolute = flat.abs()

    mean = float(flat.mean())
    std = float(flat.std(unbiased=True))
    centered = flat - mean

    # Excess kurtosis: 0 for a Gaussian, higher means heavier tails.
    if std > 0:
        kurtosis = float((centered.pow(4).mean() / (std ** 4)) - 3.0)
        skewness = float(centered.pow(3).mean() / (std ** 3))
    else:
        kurtosis = 0.0
        skewness = 0.0

    median_abs = float(absolute.median())
    max_abs = float(absolute.max())
    # Outlier ratio: how far the largest weight sits above the typical one.
    # This is the quantity the block-scaling mechanism is sensitive to.
    outlier_ratio = max_abs / median_abs if median_abs > 0 else 0.0

    q99 = float(torch.quantile(absolute.float(), 0.99))
    q999_index = max(int(n * 0.999) - 1, 0)
    sorted_abs, _ = torch.sort(absolute)
    q999 = float(sorted_abs[q999_index])

    # Per-block dynamic range: the mechanism operates within blocks of 64, so
    # the mean block-level max/mean ratio is the mechanistically-motivated
    # feature, whereas global kurtosis is only a proxy for it.
    n_blocks = n // BLOCK_SIZE
    if n_blocks > 0:
        blocks = absolute[: n_blocks * BLOCK_SIZE].reshape(n_blocks, BLOCK_SIZE)
        block_max = blocks.max(dim=1).values
        block_mean = blocks.mean(dim=1).clamp_min(1e-12)
        block_dynamic_range = float((block_max / block_mean).mean())
        block_range_p95 = float(torch.quantile((block_max / block_mean), 0.95))
    else:
        block_dynamic_range = 0.0
        block_range_p95 = 0.0

    return {
        "n_params": int(n),
        "std": std,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "max_abs": max_abs,
        "median_abs": median_abs,
        "outlier_ratio": outlier_ratio,
        "q99_abs": q99,
        "q999_abs": q999,
        "block_dynamic_range": block_dynamic_range,
        "block_range_p95": block_range_p95,
    }


def quantize_dequantize_rtn(weight: torch.Tensor, bits: int = 4,
                            block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Blockwise absmax round-to-nearest quantize-dequantize.

    Reference implementation of what bitsandbytes does for 4-bit: reshape into
    blocks, scale each by its own absmax, round to the grid, rescale. Used to
    measure reconstruction error without loading a quantized model.
    """
    flat = _to_float32(weight)
    n = flat.numel()
    pad = (-n) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])

    blocks = flat.reshape(-1, block_size)
    absmax = blocks.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
    normalized = blocks / absmax

    levels = 2 ** (bits - 1) - 1  # symmetric signed grid
    quantized = torch.round(normalized * levels).clamp(-levels - 1, levels) / levels
    reconstructed = (quantized * absmax).reshape(-1)[:n]
    return reconstructed


def nf4_levels() -> torch.Tensor:
    """The 16 NF4 quantile levels from the QLoRA paper.

    NF4 places its levels at the quantiles of a standard normal instead of
    uniformly, so it spends resolution where Gaussian weight mass actually is.
    Whether that helps depends on how Gaussian the layer is -- which is exactly
    the hypothesis under test.
    """
    return torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ], dtype=torch.float32)


def quantize_dequantize_nf4(weight: torch.Tensor,
                            block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Blockwise NF4 quantize-dequantize using the QLoRA level table."""
    flat = _to_float32(weight)
    n = flat.numel()
    pad = (-n) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])

    blocks = flat.reshape(-1, block_size)
    absmax = blocks.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
    normalized = (blocks / absmax).unsqueeze(-1)

    levels = nf4_levels().to(flat.device).view(1, 1, -1)
    nearest = torch.argmin((normalized - levels).abs(), dim=-1)
    reconstructed = (levels.view(-1)[nearest] * absmax).reshape(-1)[:n]
    return reconstructed


def layer_quantization_error(weight: torch.Tensor, method: str = "fp4") -> dict[str, float]:
    """Reconstruction error for one layer under a quantization method.

    Relative Frobenius error is the headline: it is scale-invariant, so layers
    of different magnitudes are comparable.
    """
    original = _to_float32(weight)
    if method == "nf4":
        reconstructed = quantize_dequantize_nf4(weight)
    elif method == "fp4":
        reconstructed = quantize_dequantize_rtn(weight, bits=4)
    elif method == "int8":
        reconstructed = quantize_dequantize_rtn(weight, bits=8)
    else:
        raise ValueError(f"Unknown method '{method}'")

    error = original - reconstructed
    original_norm = float(original.norm())
    return {
        "method": method,
        "mse": float(error.pow(2).mean()),
        "relative_frobenius_error": (
            float(error.norm() / original_norm) if original_norm > 0 else 0.0
        ),
        "max_abs_error": float(error.abs().max()),
    }


def profile_model_layers(model: Any, methods: tuple[str, ...] = ("fp4", "nf4", "int8"),
                         show_progress: bool = True) -> pd.DataFrame:
    """Statistics and quantization error for every linear layer in a model.

    Runs on CPU-resident FP16 weights; no inference and no eval data required.
    """
    records: list[dict[str, Any]] = []

    named_linear = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    iterator = tqdm(named_linear, desc="profiling layers") if show_progress else named_linear

    for name, module in iterator:
        weight = module.weight
        stats = layer_weight_statistics(weight)
        # Layer depth lets the analysis ask whether sensitivity varies with
        # position in the network, independent of distribution shape.
        parts = [p for p in name.split(".") if p.isdigit()]
        depth = int(parts[0]) if parts else -1

        for method in methods:
            error = layer_quantization_error(weight, method)
            records.append({
                "layer": name,
                "depth": depth,
                "layer_type": name.split(".")[-1],
                **stats,
                **error,
            })

    return pd.DataFrame(records)
