# Qwen3-8B AWQ prefix caching

GPU: NVIDIA L4, GPU-4549637f-b276-5285-adc4-0f437a786fe4, 23034 MiB, 580.126.20. vLLM 0.11.2. Concurrency 4.

| Prefix caching | Workload | Pass | Requests | Errors | p50 TTFT (ms) | p95 TTFT (ms) | Goodput (req/s) |
|---|---|---|---:|---:|---:|---:|---:|
| Off | Shared Prefix | Cold start | 48 | 0 | 866.10 | 1290.33 | 1.10 |
| Off | Shared Prefix | Repeat | 48 | 0 | 852.55 | 1216.87 | 1.11 |
| Off | Unique Prefix | Cold start | 48 | 0 | 855.91 | 1195.59 | 1.11 |
| Off | Unique Prefix | Repeat | 48 | 0 | 860.30 | 1215.46 | 1.11 |
| On | Shared Prefix | Cold start | 48 | 0 | 58.07 | 412.37 | 2.10 |
| On | Shared Prefix | Repeat | 48 | 0 | 55.91 | 67.24 | 2.14 |
| On | Unique Prefix | Cold start | 48 | 0 | 858.79 | 1329.76 | 1.10 |
| On | Unique Prefix | Repeat | 48 | 0 | 56.43 | 67.72 | 2.14 |

The cold-start pass begins with no cached entries for that workload. Shared prefixes can warm during the pass. The repeat uses the same requests, so the repeat over unique prefixes can also hit entries from the first pass.

[Reproduction and measurement definitions](../../docs/benchmark-reproduction.md)
