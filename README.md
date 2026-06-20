# ModelScope

LLM inference optimization and quality evaluation suite. Benchmarks **Llama-3.2-3B** and **Gemma-2-2B** across 8 quantization variants (FP16 / INT8 / INT4 / INT4-NF4) on a CUDA T4 GPU, then surfaces Pareto-optimal deployment configurations automatically.

## Results

| Variant | Memory (MB) | Tokens/sec | TTFT p50 (ms) | MMLU Acc | Perplexity |
|---|---|---|---|---|---|
| llama-fp16 | 6127.8 | 22.34 | 47.27 | 0.36 | - |
| llama-int8 | 3514.7 | 6.92 | 144.46 | 0.35 | - |
| **llama-int4** | **2299.4** | **13.75** | **75.43** | **0.37** | - |
| llama-int4-nf4 | 2299.4 | 14.72 | 66.10 | 0.22 | - |
| gemma2-fp16 | 23.6 | 19.51 | 49.94 | 0.23 | - |
| gemma2-int8 | 16.6 | 5.46 | 150.00 | 0.21 | - |
| gemma2-int4 | 82.6 | 13.39 | 80.00 | 0.13 | - |
| gemma2-int4-nf4 | 82.6 | 13.33 | 75.00 | 0.16 | - |

**Key finding:** `llama-int4` is Pareto-optimal -- 62% memory reduction vs FP16 with no accuracy loss (0.37 vs 0.36 MMLU). INT8 is the worst Llama tradeoff: 69% throughput drop with minimal quality benefit.

### Deployment Recommendations

| Scenario | Config | Rationale |
|---|---|---|
| Latency priority | llama-fp16 | Highest throughput at 22.34 tokens/sec |
| Memory priority | llama-int4 | 62% VRAM reduction, quality unchanged |
| Balanced | llama-int4 | Best normalized score across all three axes |

## What It Does

- **Benchmarks** throughput (tokens/sec), latency (TTFT p50/p95), and peak GPU memory across 8 variants with resume-from-checkpoint so Colab disconnects don't lose progress
- **Evaluates** quality via MMLU accuracy (100 questions, 4 subjects), output consistency (ROUGE-L pairwise variance across 5 runs), and WikiText-2 perplexity
- **Pareto analysis** finds non-dominated configs across speed, memory, and quality simultaneously and generates a 4-panel publication-quality plot
- **Deployment recommendations** normalize all three axes and score each variant for latency, memory, and balanced scenarios
- **Prometheus metrics** endpoint via FastAPI for production observability

## Stack

PyTorch, HuggingFace Transformers, bitsandbytes, accelerate, datasets, rouge-score, matplotlib, pandas, FastAPI, Prometheus

## Run It

Open [notebooks/modelscope_colab.ipynb](notebooks/modelscope_colab.ipynb) in Google Colab with a T4 GPU runtime.

The notebook:
1. Verifies GPU and HuggingFace model access
2. Writes and executes the benchmark runner (~60-90 min)
3. Writes and executes the quality eval harness (~60-90 min)
4. Merges results, computes Pareto frontier, and generates plots
5. Saves everything to Google Drive and pushes to GitHub

## Project Structure

```
modelscope/
├── benchmarks/
│   ├── runner.py          # Latency + throughput measurement
│   ├── latency.py
│   ├── memory.py
│   └── throughput.py
├── eval/
│   ├── harness.py         # MMLU + consistency + perplexity
│   ├── mmlu.py
│   ├── consistency.py
│   └── relevance.py
├── analysis/
│   ├── pareto.py          # Pareto frontier + recommendation
│   ├── visualize.py
│   └── recommend.py
├── serve/
│   ├── api.py             # FastAPI inference endpoint
│   └── metrics.py         # Prometheus metrics
├── models/
│   ├── configs.py         # MODEL_REGISTRY + quantization configs
│   ├── loader.py
│   └── generate.py
├── notebooks/
│   └── modelscope_colab.ipynb
├── tests/
│   ├── test_pareto.py
│   └── test_recommend.py
└── results/               # Generated on Colab run
    ├── benchmark_results.csv
    ├── quality_results.csv
    ├── recommendation.json
    └── plots/
        └── final_analysis.png
```
