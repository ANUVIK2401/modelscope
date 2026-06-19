from prometheus_client import Counter, Gauge

inference_requests_total = Counter(
    "inference_requests_total",
    "Total number of completed inference requests",
)
inference_ttft_ms = Gauge(
    "inference_ttft_ms",
    "Time to first token of the most recent request, in milliseconds",
)
inference_tokens_per_sec = Gauge(
    "inference_tokens_per_sec",
    "Throughput of the most recent request, in tokens per second",
)
gpu_memory_mb = Gauge(
    "gpu_memory_mb",
    "Current CUDA memory allocated, in megabytes",
)


def update_metrics(ttft_ms: float, tokens_per_sec: float, memory_mb: float) -> None:
    inference_ttft_ms.set(ttft_ms)
    inference_tokens_per_sec.set(tokens_per_sec)
    gpu_memory_mb.set(memory_mb)
