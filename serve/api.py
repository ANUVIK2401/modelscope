import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from models.generate import generate_text
from models.loader import load_model, unload_model
from serve.metrics import inference_requests_total, update_metrics

_DEFAULT_VARIANT = "llama_int4"

# Single-model cache. All fields written together under _lock.
_cache: dict[str, Any] = {
    "variant_key": None,
    "model":       None,
    "tokenizer":   None,
}
# Serializes model swaps and inference. bitsandbytes is not safe
# for concurrent CUDA use, so every critical section acquires this lock.
_lock = threading.Lock()


def _swap_variant(variant_key: str) -> None:
    """Unload the cached model (if any) and load variant_key in its place."""
    if _cache["model"] is not None:
        unload_model(_cache["model"])
    model, tokenizer, _ = load_model(variant_key)
    _cache["variant_key"] = variant_key
    _cache["model"] = model
    _cache["tokenizer"] = tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    with _lock:
        _swap_variant(_DEFAULT_VARIANT)
    yield
    with _lock:
        if _cache["model"] is not None:
            unload_model(_cache["model"])
            _cache["variant_key"] = None
            _cache["model"] = None
            _cache["tokenizer"] = None


app = FastAPI(lifespan=lifespan, title="ModelScope Inference Server")


# ── request / response schemas ───────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    variant: str = _DEFAULT_VARIANT


class GenerateResponse(BaseModel):
    text: str
    ttft_ms: float
    tokens_per_sec: float
    memory_mb: float


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str | None]:
    return {"status": "ok", "model": _cache["variant_key"]}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Sync handler — FastAPI dispatches this to a thread-pool worker,
    keeping inference off the asyncio event loop (required by bitsandbytes).
    The lock ensures only one generate() runs at a time.
    """
    with _lock:
        if request.variant != _cache["variant_key"]:
            try:
                _swap_variant(request.variant)
            except (ValueError, KeyError) as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        model = _cache["model"]
        tokenizer = _cache["tokenizer"]

        # TTFT: prefill + one decode step, consistent with latency benchmark.
        t0 = time.perf_counter()
        generate_text(model, tokenizer, request.prompt, max_new_tokens=1)
        ttft_ms = (time.perf_counter() - t0) * 1000

        # Full generation for the response payload.
        text, n_tokens, gen_time = generate_text(
            model, tokenizer, request.prompt, max_new_tokens=request.max_tokens
        )

        memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    tokens_per_sec = n_tokens / gen_time if gen_time > 0 else 0.0

    inference_requests_total.inc()
    update_metrics(ttft_ms, tokens_per_sec, memory_mb)

    return GenerateResponse(
        text=text,
        ttft_ms=round(ttft_ms, 3),
        tokens_per_sec=round(tokens_per_sec, 2),
        memory_mb=round(memory_mb, 1),
    )


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
