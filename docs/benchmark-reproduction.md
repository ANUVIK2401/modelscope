# ModelScope benchmark reproduction

## Colab T4 offline Llama baseline

The original run used the local Git archive for `52f5fe1a408d8cbead1ff9d7e04e42ef8bbccb4b` and Hugging Face model `meta-llama/Llama-3.2-3B-Instruct`. The only cached revision observed was `0cb88a4f764b7a12671c53f0838cd831a0843b95`. The [environment file](../results/t4-llama-baseline/environment.json) identifies the Python, PyTorch, CUDA, and quantization stack.

In a Colab T4 runtime, install `requirements.txt` with `transformers==4.57.6` and `bitsandbytes==0.48.2`. Set `HF_TOKEN` in Colab Secrets for the gated model. Prefetch the model into a dedicated Hugging Face cache and verify that its cached main revision matches the revision above before switching to offline mode. Run from the directory containing `benchmarks/`.

```bash
# Use a new directory per experiment and one process per variant.
export RUN_ROOT=/content/modelscope-llama-reproduction
for variant in llama_fp16 llama_nf4; do
  MODELSCOPE_ROOT="$RUN_ROOT/$variant" HF_HUB_OFFLINE=1 \
    python -u -m benchmarks.runner --variants "$variant" --no-resume
done
```

Each variant writes its own CSV, including any failure. Confirm that both have `status=ok` and matching environment hashes. The published NF4 retry ran in a fresh process on the original runtime. Generate the published comparison with `python scripts/report_llama_baseline.py`.

## Qwen3-8B HTTP streaming

The benchmark uses one CUDA GPU with BF16 capability, vLLM 0.11.2, and the pinned checkpoint and tokenizer revisions in [the benchmark runner](../benchmarks/qwen_serving.py). Both variants use BF16 activations and KV cache. The quantized variant explicitly selects AWQ Marlin.

```bash
# On the allocated A100 or L4, in an isolated Python environment.
python -m pip install 'vllm==0.11.2' --extra-index-url https://download.pytorch.org/whl/cu128
python -u -m benchmarks.qwen_serving --output ./qwen-smoke --smoke
python -u -m benchmarks.qwen_serving --output ./qwen-main
python -u -m benchmarks.qwen_serving --output ./qwen-prefix --prefix
```

The paired run uses a client on the same host as vLLM. Each variant serves the same short and long workloads at concurrency 1, 4, and 16. The published paired run uses two repeat blocks of 16 measured requests per condition. The extended BF16 run uses two repeat blocks of 48 per condition. Both models use identical prompt sequences, the same tokenizer and chat template with thinking disabled, greedy decoding, and a fixed 64-token generation length during the load test. Long prompts summarize synthetic decision records. The HTTP latency timer starts before the client sends the request and stops on the first non-whitespace content in the same stream. The client semaphore wait is outside that timer. Successful streams must finish and report token usage.

Goodput uses a 500 ms short-prompt TTFT target and a 1,500 ms long-prompt TTFT target. It counts successful requests within the target and divides by the full condition duration. Use goodput to rank configurations only when AWQ loses at most five absolute percentage points on the paired exact-match set and both runs preserve reliable streaming. The quality set contains 20 arithmetic and 20 synthetic fact-extraction prompts. It is a limited quality check for this workload. Report both repeat blocks and the sample sizes alongside the p95.

The prefix experiment uses AWQ with concurrency 4. Each server runs a shared-prefix workload and a distinct-prefix workload. The first pass begins without warmed entries for that workload. The second pass repeats its prompts. Shared entries can be reused within the first pass as the server processes requests.

Cloud Run can launch the same main run as a one-task job, using the project ID as a required argument:

```bash
# Compact matrix with 16 requests per repeat block.
bash scripts/run_qwen_cloud.sh YOUR_EXISTING_PROJECT_ID europe-west1 benchmarks/qwen_serving.py quick
# Run the prefix comparison after the main benchmark.
bash scripts/run_qwen_cloud.sh YOUR_EXISTING_PROJECT_ID europe-west1 benchmarks/qwen_serving.py prefix
```

The job uses one L4 in `europe-west1` and zero automatic retries. The main mode uses 48 requests per repeat block and a 60-minute limit. The prefix mode has a 30-minute limit. The `quick` mode uses 16 requests per repeat block, the standard Hugging Face download configuration, and a 35-minute limit. The diagnostic `retry` mode uses 32 requests and disables Xet. It saves its files to a compressed artifact encoded in numbered log entries. Filter log entries to the exact execution, then reassemble them from the repository root.

```bash
EXECUTION=YOUR_EXECUTION_ID
gcloud logging read \
  "resource.type=cloud_run_job AND labels.\"run.googleapis.com/execution_name\"=$EXECUTION AND textPayload:MODELSCOPE_ARTIFACT" \
  --limit=12000 --format='value(textPayload)' > artifact-lines.txt
python scripts/read_qwen_cloud_artifact.py artifact-lines.txt modelscope-qwen.tar.gz
tar -xzf modelscope-qwen.tar.gz
python -m scripts.report_qwen_serving modelscope-results
# For a separate prefix archive, pass --prefix.
```
