# Llama 3.2 3B offline benchmark on Colab T4

Measured on one Tesla T4 (compute capability 7.5) in the same Colab runtime. Environment hash `7b5175011d8d`. The runtime cached Hugging Face revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`.

| Metric | FP16 | NF4 | NF4 change |
|---|---:|---:|---:|
| Post-load GPU driver VRAM (MiB) | 6,194.00 | 3,126.00 | -49.5% |
| Post-load PyTorch allocated VRAM (MiB) | 6,127.80 | 2,263.90 | -63.1% |
| Batch 1 throughput (output tok/s) | 27.79 | 19.59 | -29.5% |
| Batch 4 throughput (output tok/s) | 105.39 | 45.00 | -57.3% |
| Batch 16 throughput (output tok/s) | 384.85 | 179.08 | -53.5% |
| Batch 32 throughput (output tok/s) | 620.14 | 323.23 | -47.9% |
| One-token generation p50 (ms) | 42.66 | 73.75 | +72.9% |
| One-token generation p95 (ms) | 53.46 | 88.38 | +65.3% |

FP16 delivered higher throughput at every batch size measured. NF4 reduced post-load VRAM under both memory readings. This comparison includes one successful run per variant.

The post-load driver figure is the change in device-wide used VRAM across model load. It includes allocator reservations and any other device activity during that window. The PyTorch figure is the change in live allocations tracked by its allocator. The latency rows time a separate one-token CUDA generation call after warm-up. HTTP request time, server queues, and streaming are outside that measurement. Throughput measures batches of identical prompts with a fixed generation length. No quality scores were collected for this comparison.

The initial NF4 pass stopped at the memory sanity check. The isolated retry used the cached weights in a fresh Python process and passed the same check. The magnitude of the initial reading alone cannot establish its cause.

Stack: Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0, Transformers 4.57.6, bitsandbytes 0.48.2, Accelerate 1.15.0.

[Initial CSV](initial.csv) · [Isolated NF4 CSV](isolated-nf4.csv) · [Retry log](isolated-nf4.log) · [FastAPI GPU smoke](fastapi-smoke.log) · [Environment](environment.json) · [Reproduction](../../docs/benchmark-reproduction.md)
