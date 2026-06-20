import os, gc, csv, time, torch, numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.makedirs("/content/modelscope/results", exist_ok=True)

VARIANTS = [
    {"variant": "llama-fp16", "model_id": "meta-llama/Llama-3.2-3B-Instruct", "quant_config": None},
    {"variant": "llama-int8", "model_id": "meta-llama/Llama-3.2-3B-Instruct", "quant_config": BitsAndBytesConfig(load_in_8bit=True)},
    {"variant": "llama-int4", "model_id": "meta-llama/Llama-3.2-3B-Instruct", "quant_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)},
    {"variant": "llama-int4-nf4", "model_id": "meta-llama/Llama-3.2-3B-Instruct", "quant_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)},
    {"variant": "gemma2-fp16", "model_id": "google/gemma-2-2b-it", "quant_config": None},
    {"variant": "gemma2-int8", "model_id": "google/gemma-2-2b-it", "quant_config": BitsAndBytesConfig(load_in_8bit=True)},
    {"variant": "gemma2-int4", "model_id": "google/gemma-2-2b-it", "quant_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)},
    {"variant": "gemma2-int4-nf4", "model_id": "google/gemma-2-2b-it", "quant_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)},
]

PROMPT = "Explain the attention mechanism in transformers in detail."
MAX_NEW_TOKENS = 100
RUNS = 5
CSV_PATH = "/content/modelscope/results/benchmark_results.csv"

def get_already_done():
    if not os.path.exists(CSV_PATH): return []
    with open(CSV_PATH) as f:
        return [row["variant"] for row in csv.DictReader(f)]

def write_row(row):
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists: w.writeheader()
        w.writerow(row)

def measure_variant(v):
    print("=" * 50)
    print("Loading " + v["variant"] + "...")
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / (1024**2)
    t_load = time.time()
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if v["quant_config"]:
        kwargs["quantization_config"] = v["quant_config"]
        del kwargs["torch_dtype"]
    model = AutoModelForCausalLM.from_pretrained(v["model_id"], **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(v["model_id"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    load_time = time.time() - t_load
    memory_mb = round((torch.cuda.max_memory_allocated() / (1024**2)) - mem_before, 1)
    print("Loaded in " + str(round(load_time, 1)) + "s | Memory: " + str(memory_mb) + "MB")
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    ttft_list, tps_list = [], []
    print("Running " + str(RUNS) + " benchmark iterations...")
    for _ in tqdm(range(RUNS)):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        n_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_time = t_end - t_start
        ttft_list.append((total_time / max(n_tokens, 1)) * 1000)
        tps_list.append(n_tokens / total_time)
    result = {
        "variant": v["variant"],
        "model_id": v["model_id"],
        "memory_mb": memory_mb,
        "load_time_s": round(load_time, 2),
        "ttft_p50_ms": round(float(np.percentile(ttft_list, 50)), 2),
        "ttft_p95_ms": round(float(np.percentile(ttft_list, 95)), 2),
        "tokens_per_sec": round(float(np.mean(tps_list)), 2),
    }
    print("Result: " + str(result))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result

already_done = get_already_done()
print("Already completed: " + str(already_done))
for v in tqdm(VARIANTS, desc="Overall Progress"):
    if v["variant"] in already_done:
        print("Skipping " + v["variant"] + " - already done")
        continue
    try:
        row = measure_variant(v)
        write_row(row)
        print("Saved: " + v["variant"])
    except Exception as e:
        print("ERROR on " + v["variant"] + ": " + str(e))
        gc.collect()
        torch.cuda.empty_cache()
        continue
print("All benchmarks complete!")
