"""Environment capture and seeding.

Every result row carries the exact software/hardware stack that produced it.
Without this, a reviewer cannot tell whether a number is a property of the
quantization method or of a bitsandbytes point release.
"""

import hashlib
import os
import platform
import random
import subprocess
from typing import Any

import numpy as np
import torch

SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Seed every RNG that can affect a measurement.

    Note: determinism here covers sampling and data selection. Kernel-level
    nondeterminism in matmul reductions is NOT removed (deterministic algorithms
    would change the performance being measured), so throughput still varies
    run to run. That variance is what the confidence intervals are for.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except ImportError:
        return "not-installed"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def capture_environment() -> dict[str, Any]:
    """Full provenance record. Attached to every results CSV."""
    env: dict[str, Any] = {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "accelerate": _package_version("accelerate"),
        "datasets": _package_version("datasets"),
        "seed": SEED,
    }

    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda or "unknown"
        major, minor = torch.cuda.get_device_capability(0)
        env["compute_capability"] = f"{major}.{minor}"
        env["gpu_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1
        )
        # Compute capability >= 7.5 has INT8 tensor cores; >= 8.0 (Ampere) has
        # the wider INT8/INT4 paths. T4 is 7.5 -- INT8 results there do NOT
        # generalize to Ampere, and the writeup must say so.
        env["is_ampere_or_newer"] = major >= 8
    else:
        env["gpu_name"] = "cpu"
        env["cuda"] = "none"
        env["compute_capability"] = "none"
        env["gpu_total_mb"] = 0.0
        env["is_ampere_or_newer"] = False

    return env


def environment_hash(env: dict[str, Any] | None = None) -> str:
    """Short stable hash of the stack. Rows with different hashes are not
    directly comparable and the analysis layer warns when they are mixed."""
    env = env if env is not None else capture_environment()
    keys = ("torch", "transformers", "bitsandbytes", "gpu_name", "cuda")
    payload = "|".join(f"{k}={env.get(k)}" for k in keys)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
