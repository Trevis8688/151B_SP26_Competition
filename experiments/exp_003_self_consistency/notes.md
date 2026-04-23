# Experiment: self_consistency

**Date:** 2026-04-22
**Baseline compared against:** exp_001_longer_context (Kaggle: 0.526)

## Hypothesis
Running 5 samples per question and taking the modal `\boxed{}` answer (self-consistency / majority voting) will reduce answer variance and improve accuracy, since the dominant failure bucket is wrong_math (51%) not missing_boxed — more samples per question should amplify the signal when the model knows the right answer but is noisy.

## Change from baseline
- `SamplingParams(n=5, temperature=0.7)` — 5 samples per question, slightly higher temperature for diversity
- Post-processing: extract `\boxed{}` from each of the 5 samples, take the mode; fall back to sample 0 if no majority
- Prompts: reverted to exp_001 (no enforcement wording — exp_002 proved that hurts)
- `split: dev` — validate on 200q first before committing to a full run
- `max_tokens`: 8192 (corrected from template's 16384)

## Run 1 — OOM crash (2026-04-23)

Kaggle run failed at ~5.9 hours into the dev run with:
```
RuntimeError: CUDA out of memory. Tried to allocate 380.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 241.81 MiB is free.
```

**Root causes:**
1. vLLM v0.19.1 runs the V1 engine by default and ignores `VLLM_USE_V1=0`. The V1 engine uses CUDA graph capture + Inductor compilation, adding ~1–2 GB VRAM overhead per GPU on top of model weights and KV cache.
2. `n=5` with `max_tokens=8192` produces sequences up to 8,423 tokens; with `max_num_seqs=32` and `max_num_batched_tokens=20480` the KV cache filled GPU 0 completely.
3. Memory fragmentation accumulated over 6 hours, making a 380 MiB allocation fail near the end of the run.

**Fixes applied:**
- `enforce_eager=True` added to `LLM()` — disables CUDA graph capture in V1 engine (~1–2 GB savings)
- `PYTORCH_ALLOC_CONF=expandable_segments:True` added to env block before vllm import — reduces fragmentation
- `max_num_seqs`: 32 → 16
- `max_num_batched_tokens`: 20480 → 10240
- `gpu_memory_utilization`: 0.90 → 0.85

## Dev results
_Fill in after re-running with OOM fixes._

| Metric | Baseline | This | Δ |
|--------|---------:|-----:|---:|
| Overall | | | |
| MCQ | | | |
| Free-form | | | |

## Topic movers
_Top 3 topics that improved / regressed._

## Conclusion
- [ ] Keep (merge into `main` prompt set)
- [ ] Discard
- [ ] Needs variant — next experiment idea:
