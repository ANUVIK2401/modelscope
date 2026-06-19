import os, gc, csv, torch, numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from rouge_score import rouge_scorer

os.makedirs("/content/modelscope/results", exist_ok=True)

VARIANTS = [
    {"variant": "gemma2-fp16", "model_id": "google/gemma-2-2b-it", "quant_config": None},
    {"variant": "gemma2-int8", "model_id": "google/gemma-2-2b-it", "quant_config": BitsAndBytesConfig(load_in_8bit=True)},
    {"variant": "gemma2-int4", "model_id": "google/gemma-2-2b-it", "quant_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)},
    {"variant": "gemma2-int4-nf4", "model_id": "google/gemma-2-2b-it", "quant_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)},
]

BENCH_CSV = "/content/modelscope/results/benchmark_results.csv"
EVAL_CSV = "/content/modelscope/results/quality_results.csv"
SUBJECTS = ["abstract_algebra", "anatomy", "astronomy", "computer_security"]
CONSISTENCY_PROMPT = "What are the key differences between supervised and unsupervised learning?"

def get_already_done(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [row["variant"] for row in csv.DictReader(f)]

def append_row(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def load_mmlu():
    questions = []
    for subject in SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for item in ds.select(range(min(25, len(ds)))):
            questions.append({"question": item["question"], "choices": item["choices"], "answer": item["answer"]})
    print("Loaded " + str(len(questions)) + " questions")
    return questions

def score_mmlu(model, tokenizer, questions):
    correct = 0
    for q in tqdm(questions, desc="MMLU"):
        messages = [
            {"role": "user", "content": "Question: " + q["question"] + "\nA: " + q["choices"][0] + "\nB: " + q["choices"][1] + "\nC: " + q["choices"][2] + "\nD: " + q["choices"][3] + "\nAnswer with only A, B, C, or D:"}
        ]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = "Question: " + q["question"] + "\nA: " + q["choices"][0] + "\nB: " + q["choices"][1] + "\nC: " + q["choices"][2] + "\nD: " + q["choices"][3] + "\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = decoded[0].upper() if decoded else "X"
        if pred == ["A","B","C","D"][q["answer"]]:
            correct += 1
    return round(correct / len(questions), 4)

def score_consistency(model, tokenizer):
    messages = [{"role": "user", "content": CONSISTENCY_PROMPT}]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = CONSISTENCY_PROMPT
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs_list = []
    for _ in tqdm(range(5), desc="Consistency"):
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        outputs_list.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(outputs_list[i], outputs_list[j])["rougeL"].fmeasure
              for i in range(len(outputs_list)) for j in range(i+1, len(outputs_list))]
    return round(float(np.mean(scores)), 4)

def load_model(v):
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if v["quant_config"]:
        kwargs["quantization_config"] = v["quant_config"]
        del kwargs["torch_dtype"]
    model = AutoModelForCausalLM.from_pretrained(v["model_id"], **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(v["model_id"])
    return model, tokenizer

def benchmark_variant(model, tokenizer, variant_name, model_id):
    import time
    PROMPT = "Explain the attention mechanism in transformers in detail."
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    ttft_list, tps_list = [], []
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / (1024**2)
    for _ in tqdm(range(5), desc="Benchmark"):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        n_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_time = t_end - t_start
        ttft_list.append((total_time / max(n_tokens, 1)) * 1000)
        tps_list.append(n_tokens / total_time)
    memory_mb = round((torch.cuda.max_memory_allocated() / (1024**2)) - mem_before, 1)
    return {
        "variant": variant_name,
        "model_id": model_id,
        "memory_mb": memory_mb,
        "load_time_s": 0,
        "ttft_p50_ms": round(float(np.percentile(ttft_list, 50)), 2),
        "ttft_p95_ms": round(float(np.percentile(ttft_list, 95)), 2),
        "tokens_per_sec": round(float(np.mean(tps_list)), 2),
    }

print("Loading MMLU...")
questions = load_mmlu()

bench_done = get_already_done(BENCH_CSV)
eval_done = get_already_done(EVAL_CSV)

for v in tqdm(VARIANTS, desc="Gemma-2 Variants"):
    try:
        print("\nLoading " + v["variant"] + "...")
        model, tokenizer = load_model(v)

        if v["variant"] not in bench_done:
            bench_row = benchmark_variant(model, tokenizer, v["variant"], v["model_id"])
            append_row(BENCH_CSV, bench_row)
            print("Benchmark saved: " + str(bench_row["tokens_per_sec"]) + " tok/s, " + str(bench_row["memory_mb"]) + " MB")

        if v["variant"] not in eval_done:
            mmlu_acc = score_mmlu(model, tokenizer, questions)
            consistency = score_consistency(model, tokenizer)
            append_row(EVAL_CSV, {"variant": v["variant"], "model_id": v["model_id"], "mmlu_accuracy": mmlu_acc, "consistency_score": consistency})
            print("Eval saved: MMLU=" + str(mmlu_acc) + " Consistency=" + str(consistency))

        del model
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print("ERROR on " + v["variant"] + ": " + str(e))
        continue

print("\nAll Gemma-2 variants complete!")
