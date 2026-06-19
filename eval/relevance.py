import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

_ST_MODEL_NAME = "all-MiniLM-L6-v2"
_MAX_NEW_TOKENS = 100

# Lazy singleton: loaded on first call, reused across variants.
_st_model: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        # Force CPU so the LLM and the embedding model don't compete for T4 VRAM.
        _st_model = SentenceTransformer(_ST_MODEL_NAME, device="cpu")
    return _st_model


def measure_relevance(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    reference: str,
) -> float:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][input_len:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)

    st = _get_st_model()
    # normalize_embeddings=True → dot product equals cosine similarity in [-1, 1].
    embeddings = st.encode([generated, reference], normalize_embeddings=True)
    score = float(np.dot(embeddings[0], embeddings[1]))
    return max(0.0, min(1.0, score))
