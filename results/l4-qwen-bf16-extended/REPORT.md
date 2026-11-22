# Extended Qwen3-8B BF16 run on Cloud Run L4

The BF16 variant completed all six conditions with two repeats of 48 requests each. All 576 streams succeeded. The workload lengths, decoder, and latency targets follow the [streaming protocol](../../docs/benchmark-reproduction.md).

| Workload | Concurrency | p95 TTFT across repeats (ms) | Goodput across repeats (req/s) |
|---|---:|---:|---:|
| Long | 1 | 435.7–437.5 | 0.23–0.23 |
| Long | 4 | 1196.2–1196.4 | 0.65–0.65 |
| Long | 16 | 5737.6–5999.8 | 0.74–0.77 |
| Short | 1 | 80.2–80.7 | 0.25–0.25 |
| Short | 4 | 154.4–155.4 | 0.94–0.94 |
| Short | 16 | 266.9–276.6 | 3.55–3.55 |

BF16 scored 28 out of 40 on the synthetic exact-match set. The execution reached its 30-minute task limit after the switch to AWQ. Cloud Logging retained the condition summaries. The job ended before exporting per-request samples.

[Structured condition data](conditions.csv)
