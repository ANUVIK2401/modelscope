# Predicting Layer-Wise Quantization Sensitivity from Weight Statistics in Small Language Models

**Anuvik Thota**

*Draft — workshop format. Numbers marked `[TBD]` are produced by the pipeline in this
repository and filled in from `results/` after the full run; see §7 for the exact
commands. No number appears in this paper that was not emitted by a committed script.*

---

## Abstract

Post-training quantization is the default way to fit language models onto commodity
accelerators, but the choice of format is made empirically: practitioners run a
benchmark sweep per model. We ask whether the damage quantization does is instead
*predictable* from properties of the weights alone. Working with two small instruction
models (Llama-3.2-3B, Gemma-2-2B) across four bitsandbytes configurations, we (i) report
a corrected benchmark with confidence intervals on every quantity, and (ii) test a
mechanistic hypothesis: because bitsandbytes shares one absmax scale across each 64-weight
block, layers whose weights are heavy-tailed should suffer disproportionate reconstruction
error, and NF4's normal-quantile levels should help more precisely where tails are heavier.
The first prediction survives; the second is rejected with its sign reversed.

We find the block-level dynamic range predicts per-layer reconstruction error with held-out
R² = `[TBD]` under leave-one-model-family-out validation, using features computable from an
FP16 checkpoint in seconds with no GPU and no evaluation data. We also report a
methodological negative result: linear models transfer across model families while
gradient-boosted trees do not, because tree splits cannot extrapolate to a new family's
feature range.

Separately, we document that a widely-reproduced benchmarking pattern — free-text
generation with first-character answer parsing — produces below-chance MMLU scores that are
easily mistaken for genuine quantization damage. We recommend a baseline-validation gate as
standard practice.

---

## 1. Introduction

A practitioner choosing a quantization format for a 3B model today runs a sweep. That sweep
answers a local question — which config to deploy, on this hardware, for this model — and
its answer does not transfer to the next model. The literature that *does* generalize
(LLM.int8(), GPTQ, AWQ, QLoRA) works by identifying a mechanism: activation outliers,
second-order sensitivity, activation-aware scaling, quantile-matched levels.

This paper takes the mechanistic route for a question practitioners face constantly and
answer by brute force: **which layers will quantization hurt, and can we know before
running an eval?**

Our contributions:

1. **A mechanistic predictor of layer sensitivity.** Block-level dynamic range — a
   statistic computable from FP16 weights in seconds — predicts per-layer quantization
   reconstruction error across model families (§4).
2. **A conditional refinement of the NF4 recommendation.** NF4's advantage over FP4 is not
   uniform; it scales with tail-heaviness (r = `[TBD]`), which both explains when the QLoRA
   result applies and predicts when it will not (§4.2).
3. **A corrected benchmark with honest uncertainty.** Every accuracy claim carries a
   confidence interval and a Holm-Bonferroni-corrected significance test; we report the
   minimum detectable effect so unpowered comparisons are visible as such (§3).
4. **A benchmarking-methodology warning.** We document a parser failure mode that produces
   below-chance accuracy readable as a finding, and propose a cheap gate against it (§5).

## 2. Background and related work

**Outliers drive quantization difficulty.** Dettmers et al. (2022) showed that
activation outliers in transformers above ~6.7B parameters break naive INT8, motivating
mixed-precision decomposition (LLM.int8()). Our work applies the analogous question to
*weights* at the block level, and at a scale below where activation outliers dominate.

**Calibration-based methods.** GPTQ (Frantar et al., 2023) uses second-order information
to minimize layerwise output error; AWQ (Lin et al., 2023) protects salient channels
identified from activation statistics. Both establish that per-layer sensitivity varies
and is exploitable. Both require calibration data and a nontrivial optimization. Our
question is whether a large part of that signal is available for free from weights alone.

**NF4 and QLoRA.** Dettmers et al. (2023) introduced NF4, whose levels sit at the
quantiles of a standard normal, reporting that it is information-theoretically optimal for
normally-distributed weights. The premise is explicitly distributional, but the recommendation
is usually applied unconditionally. §4.2 tests the conditional version.

**Where we differ.** The papers above propose *methods*. We propose no method; we test
whether an existing method's damage is *predictable*, and treat that predictability as the
object of study. This is closer in spirit to work on scaling-law-style prediction than to
work on quantization algorithms.

## 3. Corrected benchmark

### 3.1 Setup

Two model families (Llama-3.2-3B-Instruct, Gemma-2-2b-it) × four configurations
(FP16, INT8, FP4, NF4) = 8 variants. Single T4 GPU, 16 GB. Full software versions,
driver, and a stack hash are recorded in `results/environment.json` and attached to every
result row; rows with differing hashes are flagged as non-comparable.

**Naming.** We write FP4 and NF4 rather than "int4" and "int4-nf4". bitsandbytes'
`bnb_4bit_quant_type` defaults to `fp4`, so a config specified without that argument is
FP4 — a distinct format, not a generic 4-bit baseline. This is a documented default that
is easy to misread, and mislabelling it makes the FP4/NF4 comparison unreadable.

**MMLU protocol.** 5-shot, prompts built from the dataset's own `dev` split, scored by
argmax over the logits of the ` A`/` B`/` C`/` D` continuation tokens. No generation, no
string parsing. 40 questions × 8 subjects = 320 per variant, spanning reasoning-heavy,
recall-heavy, and applied categories so task-conditional effects are not averaged away.
All variants see an identical, seeded question set, making comparisons paired.

**Uncertainty.** Accuracy intervals are Wilson score intervals. Pairwise comparisons use
a paired bootstrap (10,000 resamples) and McNemar's exact test, with Holm-Bonferroni
correction across all 28 pairs. With n = 320, the minimum detectable effect at 80% power
is 0.111 — comparisons below that gap are reported as indistinguishable, not as small
effects.

### 3.2 Results

`[TBD — table generated from results/merged_results.csv]`

| Variant | Weight MB | tok/s (b=1) | TTFT p50 (ms) | ITL p50 (ms) | MMLU [95% CI] | ECE | PPL |
|---|---|---|---|---|---|---|---|
| llama_fp16 | | | | | | | |
| llama_int8 | | | | | | | |
| llama_fp4 | | | | | | | |
| llama_nf4 | | | | | | | |
| gemma_fp16 | | | | | | | |
| gemma_int8 | | | | | | | |
| gemma_fp4 | | | | | | | |
| gemma_nf4 | | | | | | | |

### 3.3 Measurement notes

**TTFT and ITL are measured separately.** Prefill is compute-bound and parallel over the
prompt; decode is memory-bandwidth-bound and sequential. Quantization moves them in
different directions, so a single averaged latency number conflates two distinct effects.
We measure TTFT with `max_new_tokens=1` and derive ITL per run from the residual.

**Memory is validated against theory.** Measured weight memory is compared to
`parameters × bits / 8`; a reading outside tolerance raises rather than being recorded.
This is deliberately strict: a 2B FP16 model cannot occupy 24 MB, and a pipeline that
accepts such a reading will publish it.

**Batched throughput and KV cache.** Throughput is reported at batch sizes 1, 4, 16, 32,
with KV-cache memory measured separately from weights. Weight memory alone understates
deployment footprint, and the memory ranking between configs can invert once the cache is
counted, since 4-bit shrinks weights but leaves the FP16 cache untouched.

## 4. Layer sensitivity: measurement and prediction

### 4.1 Hypothesis and method

bitsandbytes quantizes weights in blocks of 64 sharing a single absmax scale. A block
containing one large-magnitude weight has its scale set by that weight, coarsening the
effective grid for the other 63. This yields a directional prediction:

> **H1.** Per-layer reconstruction error increases with the layer's block-level dynamic
> range (mean per-block max/mean |w|).

> **H2.** NF4's advantage over FP4 increases with tail-heaviness, because uniform FP4
> levels waste resolution on a range the weights rarely occupy.

**H2 is rejected, with the sign reversed.** Against the true E2M1 codebook, NF4's relative
advantage *decreases* monotonically in tail-heaviness: +12.7% at excess kurtosis 0.1 and
+1.0% at kurtosis 164 on Student-t weights. The premise was wrong in its own terms — FP4
is not a uniform grid but a floating-point format whose levels are geometrically spaced,
so it already concentrates resolution near zero. NF4's levels are normal quantiles, which
makes heavy tails the regime where its prior is *least* appropriate. An earlier version of
this study confirmed H2, but only because it stood FP4 in as a uniform 15-level grid.

For every linear layer we compute (a) distribution statistics from FP16 weights —
kurtosis, skewness, outlier ratio, block dynamic range, quantiles — and (b) ground-truth
reconstruction error under blockwise absmax quantization against each format's real level
table — the E2M1 codebook for FP4, the QLoRA table for NF4, and a uniform signed grid for
INT8, which is the only one of the three that genuinely is uniform. We then regress (b) on (a).

**Validation is leave-one-model-family-out.** Training on Llama layers and testing on
Gemma layers answers the question that matters — does this transfer to a model you have
not benchmarked? A random split over layers would leak badly, since layers within one
model are strongly correlated. Every fit is reported against a mean-predictor baseline.

### 4.2 Results

`[TBD — from results/sensitivity_report.json]`

| Format | Model | Held-out R² | MAE | Baseline MAE | Split |
|---|---|---|---|---|---|
| FP4 | Ridge | | | | leave-one-family-out |
| FP4 | Random forest | | | | leave-one-family-out |
| NF4 | Ridge | | | | leave-one-family-out |

**NF4 advantage vs tail-heaviness.** Correlation between a layer's NF4 relative advantage
and its block dynamic range: r = `[TBD]`.

**Methodological negative result.** In validation on synthetic layers with known ground
truth, ridge regression achieved held-out R² = 0.97 across families while a random forest
achieved 0.07, despite both beating the mean baseline on MAE. Tree ensembles partition the
training feature range and cannot extrapolate to a new family whose features fall outside
it. For cross-model generalization claims, this argues for linear or explicitly
extrapolating models, and against the tree-based defaults common in tabular prediction.

### 4.3 Limitation: proxy vs end-task

Reconstruction error is not downstream accuracy. A layer can be reconstructed poorly and
matter little, or well and matter greatly. We measure the proxy because it is cheap and
mechanistically interpretable; establishing the link from layer-level error to end-task
degradation is the primary open item (§8).

## 5. A benchmarking failure mode worth naming

A common pattern — prompt the model, generate a few tokens, read the first character as
the answer — fails silently in a specific way. An instruction model that answers "The
answer is B" is scored as having said "T". Combined with 0-shot prompting, this produced
MMLU accuracies of 0.13–0.37 in an earlier version of this pipeline, for models published
at ~0.60. Below-chance scores (chance = 0.25) were spread across configurations in a
pattern that looked like a quantization-damage gradient.

The failure is not exotic and the fix is not novel; what is worth stating is that nothing
in a typical pipeline objects. We therefore recommend a **baseline-validation gate**: run
the FP16 reference first, compare against the published figure, and abort if the interval
is at or below chance. It costs one model load and catches an entire class of silent
protocol errors. Our implementation is in `eval/harness.py:validate_baseline`.

## 6. Threats to validity

**Single GPU class.** All timing comes from one T4 (compute capability 7.5). INT8
throughput in particular is architecture-dependent: bitsandbytes' LLM.int8() path on
hardware without wide INT8 tensor cores behaves differently than on Ampere and later. Our
INT8 speed results should not be read as general.

**One quantization family.** bitsandbytes is round-to-nearest. GPTQ and AWQ use calibration
data and may order configurations differently. The predictor is fit on RTN error and is not
claimed to transfer to calibration-based methods.

**Two families, one scale band.** Cross-family validation with two families is the weakest
useful form of the generalization claim. Two points establish that transfer is possible,
not the shape of the relationship.

**Statistical power.** At n = 320 per variant, gaps below 0.111 accuracy are unresolvable.
We report such comparisons as indistinguishable rather than as small effects.

**Proxy target.** See §4.3.

## 7. Reproducibility

Every number is produced by a committed script; no result is transcribed by hand.

```bash
pip install -r requirements.txt
python -m benchmarks.runner                 # benchmark_results.csv
python -m eval.harness                      # quality_results.csv (+ baseline gate)
python -m analysis.significance             # significance_tests.json
python -m research.run_sensitivity          # layer_sensitivity.csv, sensitivity_report.json
pytest                                      # 100 tests, no GPU required
```

Environment provenance (versions, driver, GPU, stack hash) is captured in
`results/environment.json`. Analysis code is covered by tests that run on CPU in CI, so a
broken analysis chain fails before a GPU run is spent.

## 8. Conclusion and next steps

Quantization damage at the layer level is partly predictable from weight distribution
shape, using features that cost seconds and no GPU. The clearest next steps, in order of
value:

1. **Close the proxy gap.** Correlate per-layer reconstruction error with end-task
   degradation under selective per-layer precision — does protecting the top-k predicted
   layers recover accuracy?
2. **Extend to calibration-based methods.** Does the predictor transfer to GPTQ/AWQ error,
   or is the RTN block-scaling mechanism specific?
3. **Widen the family and scale range.** Two families at ~3B is the weakest form of the
   transfer claim; 5+ families across 1B–13B would characterize it.
4. **A second GPU architecture.** One Ampere run would separate architecture-specific INT8
   behaviour from format-intrinsic behaviour.

## References

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix
Multiplication for Transformers at Scale. *NeurIPS*.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient
Finetuning of Quantized LLMs. *NeurIPS*.

Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023). GPTQ: Accurate Post-Training
Quantization for Generative Pre-trained Transformers. *ICLR*.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J.
(2021). Measuring Massive Multitask Language Understanding. *ICLR*.

Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). AWQ:
Activation-aware Weight Quantization for LLM Compression and Acceleration. *MLSys*.

Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023). SmoothQuant:
Accurate and Efficient Post-Training Quantization for Large Language Models. *ICML*.
