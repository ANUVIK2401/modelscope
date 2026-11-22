# Contributing

This repository has one organizing rule, and it exists because an earlier
version of this project published findings that were wrong:

> **Every number in the writeup is emitted by a committed, tested script.**

If a contribution breaks that chain, it will not be merged, however correct the
number happens to be.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
pytest                       # 113 tests, CPU-only, no GPU required
ruff check . --exclude venv
```

The full test suite runs without a GPU by design. If you find yourself needing
one to test a change, the change is probably doing measurement work that belongs
behind an interface you can fake.

---

## The rules that encode past failures

Each of these exists because violating it produced a published bad number. They
are listed in `CLAUDE.md` too; this is the short version.

**Measurement**

- Never report an accuracy without a confidence interval. At n=320 the minimum
  detectable effect is 0.111 — smaller gaps are noise and must be reported as
  indistinguishable, not as small effects.
- Never measure GPU memory with `max_memory_allocated()` deltas alone. Under
  `device_map="auto"` that pattern reported 24 MB for a 2B FP16 model. Use
  `models/loader.py:measure_load_memory()`, which reads the driver and asserts
  against `params × bits / 8`.
- Never score MMLU by parsing generated text. Logit-score over the ` A`/` B`/
  ` C`/` D` tokens. Text parsing produced below-chance accuracy.

**Failure handling**

- Never let a variant fail silently. `except Exception: continue` hid failures
  and corrupted the Pareto frontier. Record `status="failed"` and the error.
- Never add a results column that no module produces. `recommend.py` once read a
  `relevance_score` that nothing emitted, and raised on real data.

**Code paths**

- Never write code as a string literal from a notebook cell. The notebook
  imports tested modules and nothing else. Every published bad number in this
  project's history came from the untested string-literal path.
- All output paths come from `analysis/paths.py`. Never hardcode `/content/...`.
- Naming: a 4-bit variant is `fp4` or `nf4`, never `int4`. bitsandbytes defaults
  `bnb_4bit_quant_type` to `"fp4"`, so an unqualified name hides which of two
  distinct formats is meant.

---

## Adding a quantization variant

1. Add it to `MODEL_REGISTRY` in `models/configs.py`. Nowhere else — never
   define a `BitsAndBytesConfig` inline.
2. Add the expected theoretical memory so `loader.py`'s assertion has something
   to check against.
3. Add a case to `tests/test_configs.py`.
4. Run the notebook. Commit the regenerated `results/` CSVs alongside the code.

## Adding a metric

1. Implement it in `eval/` or `benchmarks/` as a pure function over a loaded
   model, returning a dict.
2. Give it a test with a fake or tiny model. It must pass on CPU.
3. Wire it into the corresponding runner, writing incrementally — Colab
   disconnects, and a run must survive losing its last variant.
4. Add the column to `scripts/build_report.py` so it reaches the report.

---

## Changing a published finding

If new evidence contradicts something the README claims, **correct the record;
do not quietly revert.** Add a row to the retraction table in `README.md`
stating the defect, its consequence, and the fix. The table is the most
load-bearing part of this repository — it is the reason the remaining numbers
are worth reading.

---

## Pull requests

- `pytest` and `ruff check` pass. CI runs both.
- New behaviour has a test. Line coverage is **64.5%**, and CI fails below a 60% floor.

  That number is lower than it looks. The suite is CPU-only by design, so modules
  that only execute against a real GPU sit well under it — `models/loader.py`
  (28%), `models/env.py` (29%), `eval/perplexity.py` (21%). The pure-logic
  modules that carry the findings are covered far better: `analysis/statistics.py`
  93%, `research/predictor.py` 94%, `research/sensitivity.py` 85%,
  `models/configs.py` 100%. Raise the gate when you add coverage; never lower it.
- Commit messages follow `type: description` (`feat`, `fix`, `refactor`, `docs`,
  `test`, `chore`, `perf`, `ci`).
- If the change affects a number, include the regenerated `results/` artifact and
  say which run produced it. Rows carry an `env_hash`; comparing across hashes
  is not valid.
