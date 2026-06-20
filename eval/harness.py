import os, gc, csv, math, torch, numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from rouge_score import rouge_scorer

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

CSV_PATH = "/content/modelscope/results/quality_results.csv"
SUBJECTS = ["abstract_algebra", "anatomy", "astronomy", "computer_security"]
CONSISTENCY_PROMPT = "What are the key differences between supervised and unsupervised learning?"

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

def load_model(v):
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if v["quant_config"]:
        kwargs["quantization_config"] = v["quant_config"]
        del kwargs["torch_dtype"]
    model = AutoModelForCausalLM.from_pretrained(v["model_id"], **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(v["model_id"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_mmlu():
    questions = []
    for subject in SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for item in ds.select(range(min(25, len(ds)))):
            questions.append({"question": item["question"], "choices": item["choices"], "answer": item["answer"]})
    print("Loaded " + str(len(questions)) + " MMLU questions")
    return questions

def score_mmlu(model, tokenizer, questions):
    correct = 0
    for q in tqdm(questions, desc="MMLU"):
        content = "Question: " + q["question"] + "\nA: " + q["choices"][0] + "\nB: " + q["choices"][1] + "\nC: " + q["choices"][2] + "\nD: " + q["choices"][3] + "\nAnswer with A, B, C, or D only:"
        messages = [{"role": "user", "content": content}]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = content
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = decoded[0].upper() if decoded else "X"
        if pred == ["A", "B", "C", "D"][q["answer"]]: correct += 1
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

def measure_perplexity(model, tokenizer):
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 50][:80]
        total_loss, total_tokens = 0.0, 0
        model.eval()
        for text in tqdm(texts, desc="Perplexity"):
            enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to("cuda")
            if enc.input_ids.shape[1] < 2: continue
            with torch.no_grad():
                loss = model(**enc, labels=enc.input_ids).loss
            total_loss += loss.item() * (enc.input_ids.shape[1] - 1)
            total_tokens += enc.input_ids.shape[1] - 1
        if total_tokens == 0: return None
        return round(math.exp(total_loss / total_tokens), 2)
    except Exception as e:
        print("Perplexity error: " + str(e))
        return None

print("Loading MMLU questions...")
questions = load_mmlu()

already_done = get_already_done()
print("Already completed: " + str(already_done))

for v in tqdm(VARIANTS, desc="Overall Eval Progress"):
    if v["variant"] in already_done:
        print("Skipping " + v["variant"])
        continue
    try:
        print("\nEvaluating " + v["variant"] + "...")
        model, tokenizer = load_model(v)
        mmlu_acc = score_mmlu(model, tokenizer, questions)
        print("MMLU: " + str(mmlu_acc))
        consistency = score_consistency(model, tokenizer)
        print("Consistency: " + str(consistency))
        perplexity = measure_perplexity(model, tokenizer)
        print("Perplexity: " + str(perplexity))
        write_row({
            "variant": v["variant"],
            "model_id": v["model_id"],
            "mmlu_accuracy": mmlu_acc,
            "consistency_score": consistency,
            "perplexity": perplexity,
        })
        print("Saved: " + v["variant"])
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print("ERROR on " + v["variant"] + ": " + str(e))
        gc.collect()
        torch.cuda.empty_cache()
        continue

print("All evals complete!")
