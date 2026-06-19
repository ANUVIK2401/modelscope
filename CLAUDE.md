# ModelScope - CLAUDE.md

## What This Project Is
LLM inference optimization and quality evaluation suite.
Benchmarks Llama-3.2-3B and Gemma-3-4b across 4 quantization 
configs each (8 total variants). Outputs Pareto-optimal deployment 
recommendations across speed, memory, and quality axes.

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
- google/gemma-3-4b-it
Both loaded via AutoModelForCausalLM + AutoTokenizer

## 4 Quantization Configs Per Model (8 Total Variants)
- fp16: torch_dtype=torch.float16, no quantization
- int8: BitsAndBytesConfig(load_in_8bit=True)
- int4: BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
- int4_nf4: BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)

## Memory Measurement Pattern
torch.cuda.reset_peak_memory_stats() before model load
torch.cuda.max_memory_allocated() / (1024**2) after load
Always del model + gc.collect() + torch.cuda.empty_cache() after each variant

## Benchmark Settings
- Batch sizes: [1, 4, 16, 32]
- Runs per measurement: 5 (report p50 and p95)
- Max new tokens per run: 100
- Standard prompt: "Explain the attention mechanism in transformers."

## Eval Settings
- MMLU: 100 questions, 4 subjects (25 each): 
  high_school_mathematics, world_history, 
  computer_science, high_school_biology
- Consistency: 5 runs same prompt, measure ROUGE-L std deviation
- Relevance: BERTScore F1 via sentence-transformers, 
  compare output vs reference answer

## Output Files
- results/benchmark_results.csv
- results/quality_results.csv  
- results/plots/memory_vs_throughput.png
- results/plots/quality_vs_speed.png
- results/plots/pareto_frontier.png
- results/recommendation.json

## Hard Rules
- Never load two models at the same time
- Always del model + empty_cache() between variants
- Write CSV incrementally after each variant (Colab disconnects)
- Never mock or hardcode benchmark numbers
- Never use asyncio for inference (not thread-safe with bitsandbytes)
- All file paths relative to project root
- Use tqdm on every loop that runs inference