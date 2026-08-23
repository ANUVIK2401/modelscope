# ModelScope - CLAUDE.md

## What This Project Is
Two halves:
1. Corrected inference benchmark: Llama-3.2-3B and Gemma-2-2B across 4
   quantization configs each (8 variants), with confidence intervals on every
   number and Holm-corrected significance tests.
2. Research contribution: predicting per-layer quantization error from FP16
   weight statistics alone, validated leave-one-model-family-out.

The research half is the point. The benchmark half exists to be correct, not to
be the contribution.

## Runtime Environment
Google Colab T4 GPU (16GB VRAM). CUDA only.
Code is written in VSCode, executed on Colab via GitHub.

## Tech Stack
- transformers + bitsandbytes: model loading + quantization
- torch (CUDA): tensor ops + memory profiling
- accelerate: device mapping
- fastapi + uvicorn: inference serving layer
- prometheus-client: metrics endpoint
- sentence-transformers: BERTScore relevance eval
- datasets: MMLU loading from HuggingFace (cais/mmlu)
- rouge-score: consistency measurement
- matplotlib + pandas: plotting + analysis
- tqdm: progress bars everywhere

## Models
- meta-llama/Llama-3.2-3B-Instruct
- google/gemma-2-2b-it
Both loaded via models/loader.py (validated memory measurement), never
AutoModelForCausalLM directly.

## 4 Quantization Configs Per Model (8 Total Variants)
Defined once in models/configs.py MODEL_REGISTRY. Never redefine inline.
- fp16, int8, fp4, nf4

CRITICAL NAMING: bitsandbytes defaults bnb_4bit_quant_type to "fp4", NOT "nf4".
A config without that argument is FP4. Never name a variant plain "int4" -- it
hides which of two distinct 4-bit formats is meant.

## Memory Measurement Pattern
Use models/loader.py measure_load_memory(). It measures via
torch.cuda.mem_get_info() (driver-level, sees allocations the caching allocator
misses under device_map) and ASSERTS the result against params x bits / 8.

Do NOT measure with max_memory_allocated() deltas alone. That pattern reported
24 MB for a 2B FP16 model in an earlier version, and a published finding was
built on the bad number.

## Benchmark Settings
- Batch sizes: [1, 4, 16, 32]
- Runs per measurement: 5 (report p50 and p95)
- Max new tokens per run: 100
- Standard prompt: "Explain the attention mechanism in transformers."

## Eval Settings
- MMLU: 5-shot, LOGIT-SCORED over " A"/" B"/" C"/" D" tokens. Never parse
  generated text -- that produced below-chance accuracy previously.
  320 questions, 8 subjects tagged reasoning/recall/applied.
- Consistency: mean pairwise ROUGE-L across 5 SAMPLED generations. Greedy
  decoding makes this metric constant by construction.
- Perplexity: sliding window (2048 ctx, 1024 stride) on WikiText-2.
- Calibration: ECE over confidence bins.
- Every variant sees the identical seeded question set (paired design).

## Output Files
Paths come from analysis/paths.py. Never hardcode /content/... anywhere.
- results/benchmark_results.csv, quality_results.csv, merged_results.csv
- results/significance_tests.json, per_question_flags.json
- results/layer_sensitivity.csv, sensitivity_report.json
- results/environment.json (provenance; every row carries its env_hash)
- results/recommendation.json, REPORT.md, plots/final_analysis.png

Results CSVs are COMMITTED. A reviewer must inspect any number without a GPU.

## Hard Rules
- Never load two models at the same time
- Always unload_model() between variants
- Write CSV incrementally after each variant (Colab disconnects)
- Never mock or hardcode benchmark numbers
- Never use asyncio for inference (not thread-safe with bitsandbytes)
- All paths via analysis/paths.py; never hardcode /content/...
- Use tqdm on every loop that runs inference

## Hard Rules (correctness -- these encode past failures)
- NEVER report an accuracy without a confidence interval. At n=320 the minimum
  detectable effect is 0.111; smaller gaps are noise and must be reported as
  indistinguishable.
- NEVER let a failed variant pass silently. Record status="failed" plus the
  error. `except Exception: continue` hid failures and corrupted the frontier.
- NEVER write code as a string literal from a notebook cell. The notebook
  imports tested modules. Every published bad number came from the untested
  string-literal path.
- ALWAYS validate the FP16 baseline against published MMLU before trusting any
  downstream comparison. Abort on failure.
- NEVER add a results column that no module produces (recommend.py read a
  phantom relevance_score and raised on real data).
- Correct the record rather than quietly reverting: retracted findings are
  documented in the README table.