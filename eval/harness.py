import os, gc, csv, torch, numpy as np
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
    {"variant": "gemma-fp16", "model_id": "google/gemma-3-4b-it", "quant_config": None},
    {"variant": "gemma-int8", "model_id": "google/gemma-3-4b-it", "quant_config": BitsAndBytesConfig(load_in_8bit=True)},
    {"variant": "gemma-int4", "model_id": "google/gemma-3-4b-it", "quant_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)},
    {"variant": "gemma-int4-nf4", "model_id": "google/gemma-3-4b-it", "quant_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)},
]

CSV_PATH = "/content/modelscope/results/quality_results.csv"
SUBJECTS = ["abstract_algebra", "anatomy", "astronomy", "computer_security"]
CONSISTENCY_PROMPT = "What are the key differences between supervised and unsupervised learning?"

def get_already_done():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH) as f:
        return [row["variant"] for row in csv.DictReader(f)]

def write_row(row):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
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
    print("Loaded " + str(len(questions)) + " questions from " + str(len(SUBJECTS)) + " subjects")
    return questions

def score_mmlu(model, tokenizer, questions):
    correct = 0
    for q in tqdm(questions, desc="MMLU"):
        prompt = "Question: " + q["question"] + "\nA: " + q["choices"][0] + "\nB: " + q["choices"][1] + "\nC: " + q["choices"][2] + "\nD: " + q["choices"][3] + "\nAnswer with A, B, C, or D only:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = decoded[0].upper() if decoded else "X"
        if pred == ["A","B","C","D"][q["answer"]]:
            correct += 1
    return round(correct / len(questions), 4)

def score_consistency(model, tokenizer):
    inputs = tokenizer(CONSISTENCY_PROMPT, return_tensors="pt").to("cuda")
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
    return AutoModelForCausalLM.from_pretrained(v["model_id"], **kwargs), AutoTokenizer.from_pretrained(v["model_id"])

print("Loading MMLU questions...")
questions = load_mmlu()

already_done = get_already_done()
print("Already completed: " + str(already_done))

for v in tqdm(VARIANTS, desc="Overall Eval Progress"):
    if v["variant"] in already_done:
        print("Skipping " + v["variant"])
        continue
    try:
        print("Evaluating " + v["variant"] + "...")
        model, tokenizer = load_model(v)
        mmlu_acc = score_mmlu(model, tokenizer, questions)
        consistency = score_consistency(model, tokenizer)
        write_row({"variant": v["variant"], "model_id": v["model_id"], "mmlu_accuracy": mmlu_acc, "consistency_score": consistency})
        print("Saved: " + v["variant"] + " | MMLU: " + str(mmlu_acc) + " | Consistency: " + str(consistency))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print("ERROR on " + v["variant"] + ": " + str(e))
        continue

print("All evals complete!")
