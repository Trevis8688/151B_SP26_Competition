# Experiment: grpo

**Started:** 2026-04-28
**Status (as of 2026-05-04):** Phase 1 complete, Phase 1B in progress, Phase 2 (training) pending
**Baseline:** exp_004_fewshot_prompts (local 55.33%, Kaggle 0.551)

## Hypothesis (unchanged)

GRPO fine-tuning on Qwen3-4B-Thinking-2507 using public.jsonl as reward signal will push accuracy beyond the prompt-engineering ceiling (~55%) by training the model to discover correct reasoning paths. Unlike SFT (which only imitates correct solutions), GRPO generates multiple responses per question and reinforces the ones that produce correct `\boxed{}` answers, allowing improvement on problems the base model currently fails.

**Key risk (materialized):** Sparse reward — with a weak base policy, most generation batches turn out uniformly wrong (or uniformly right on easy problems), producing near-zero advantage and zero gradient. The original mitigations (8 generations, format bonus) were insufficient. Solution: pre-sample to characterize per-prompt difficulty, then train only on prompts where the model gets some right and some wrong (sweet-spot curriculum filtering).

## Plan evolution

### Original plan (2026-04-28)
- One-shot GRPO via unsloth on full ~926 prompts
- 8 generations per prompt, 4096 max completion
- Train end-to-end in a single Colab session

### Current plan (2026-05-03+)
1. **Phase 1** ✅ DONE (2026-05-03): Pre-sample base model on all 926 prompts × 4 samples to characterize per-prompt difficulty (vLLM, A100, ~1.5 hr)
2. **Phase 1B** 🔄 IN PROGRESS (2026-05-04): Resample 261 length-clipped prompts at `max_tokens=8192` to recover them (~45-60 min A100)
3. **Phase 2** 📋 PENDING: GRPO training filtered to sweet-spot prompts only

### Why the pivot
Three independent GRPO gotchas surfaced during pilot debugging — all silent failures producing zero learning:

1. **LoRA dropout corrupts rollouts.** TRL+PEFT doesn't reliably switch to eval mode during generation. With `lora_dropout > 0`, sampled completions are repetitive garbage (observed: `"ationsationsations..."`). **Fix:** `lora_dropout=0.0` always for GRPO.

2. **Train/eval mode mismatch.** Even with `lora_dropout=0`, the model staying in train mode disables KV caching during generation (`use_cache=True is incompatible with gradient checkpointing` warning) → ~5× slower generation. **Fix:** monkey-patch `model.generate` to bracket with `model.eval()` / `model.train()`.

3. **Sparse reward → loss=0.** With binary correctness (0 or 1) and binary format (0 or 0.1), all 4 completions per prompt often get identical rewards → group-relative advantage = 0 → loss = 0 → no update. **Fix (partial):** granular 4-component format reward (max 0.225) for stochastic variance. **Fix (structural):** curriculum filtering — train only on prompts where rewards naturally vary across samples.

**Also discovered:** Vanilla HF generation is too slow for the full GRPO loop (~58 hr/epoch on 926 prompts). vLLM works for standalone Phase 1 inference (with the Colab stdout patch) but TRL+vLLM integration was abandoned due to brittle install chain.

## Phase 1 results (curriculum sampling, 2026-05-03)

**Setup:** Qwen3-4B-Thinking-2507 base, 926 prompts (public − dev), 4 samples per prompt at `T=1.0`, `max_tokens=4096`. vLLM on A100 40GB.

**Difficulty distribution (correct / 4):**

| Bucket | Count | % | Notes |
|---|---:|---:|---|
| 0/4 | 463 | 50.0% | Always wrong — but 261 are length-clipped |
| 1/4 | 48 | 5.2% | Sweet spot |
| 2/4 | 41 | 4.4% | Sweet spot |
| 3/4 | 58 | 6.3% | Sweet spot |
| 4/4 | 316 | 34.1% | Always right |
| **Sweet (1–3)** | **147** | **15.9%** | **Usable for GRPO** |

**Critical clipping breakdown** (of 463 uniform-0 prompts):
- 261 prompts (56%): **all 4 completions clipped** at 4096 tokens — likely solvable with longer budget → Phase 1B target
- 56 prompts (12%): mixed clipping — borderline
- 146 prompts (32%): no clipping — genuinely too hard for base model

**By question type:**
- MCQ (275 prompts): 52 sweet (19%), 159 always-wrong (58%), 64 always-right (23%)
- Free-form (651 prompts): 95 sweet (15%), 304 always-wrong (47%), 252 always-right (39%)

**Mean per-sample accuracy:** 42.3% (vs dev's 55.33% — gap is `T=1.0` exploration vs `T=0.6` eval).

## Phase 1B (in progress, 2026-05-04)

**Goal:** Resample the 261 length-clipped prompts at `max_tokens=8192` to recover legitimately-solvable ones.

**Setup:** Same vLLM/Qwen3 config as Phase 1, but:
- `MAX_COMPLETION_LEN=8192` (2× Phase 1)
- `VLLM_MAX_NUM_SEQS=16` (halved from 32 to fit 2× KV cache on 40GB)
- **NO judging in the notebook** — Phase 1's inline judging hung overnight on pathological LaTeX (sympy infinite loop). Phase 1B dumps raw vLLM outputs and is scored locally afterward via `scripts/score_raw_outputs.py` (per-call SIGALRM timeout = 15 sec).

**Expected outcome:** Recover ~100–150 additional sweet-spot prompts → ~250–280 total training set.

## Phase 2: GRPO training (planned, after Phase 1B)

Train on sweet-spot IDs only (147 from Phase 1 + recovered prompts from Phase 1B). With ~250 prompts (vs original 926), each epoch is ~3× shorter and every batch contributes gradient (no wasted uniform-reward steps). Plan: 1–3 epochs depending on reward trajectory.

## Critical learnings (saved to user memory)

These are the GRPO+TRL+PEFT gotchas that wasted ~2 days of debugging:
1. `lora_dropout=0.0` is mandatory — non-zero dropout corrupts rollouts under train mode
2. Monkey-patch `model.generate` to bracket with `eval()`/`train()` — TRL doesn't switch reliably
3. Loss=0 in GRPO usually means uniform-reward batches, not a code bug — diagnose via reward components in a `TrainerCallback`, not the loss column
4. vLLM on Colab needs `VLLM_USE_V1=0`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, AND a stdout/stderr `fileno()` patch BEFORE `import vllm`

## File structure

```
experiments/exp_009_grpo/
├── notes.md                       This file
├── config.json                    Original config (legacy from initial plan)
├── prompts.py                     System prompts + few-shot examples (exp_004 config)
│
├── train_grpo.ipynb               Phase 2: GRPO training (vanilla HF/TRL/PEFT, no vllm)
├── sample_difficulty.ipynb        Phase 1: difficulty sampling (vLLM, 4096 tokens, inline judging)
├── sample_difficulty_long.ipynb   Phase 1B: resample length-clipped at 8192 tokens (no judging)
│
├── sweet_spot_ids.json            147 sweet-spot prompt IDs from Phase 1 (1 ≤ correct ≤ 3 / 4)
├── length_clipped_ids.json        261 IDs to resample in Phase 1B (uniform-0, all clipped)
│
├── difficulty_samples.jsonl       (gitignored, ~30MB) Phase 1 scored output, 926 records
├── raw_outputs.jsonl              (gitignored, ~30MB) Phase 1 raw vLLM completions backup
├── raw_outputs_long.jsonl         (gitignored) Phase 1B raw outputs — produced after Cell 7
└── difficulty_samples_long.jsonl  (gitignored) Phase 1B scored locally
```

Related (outside the experiment folder):
- `scripts/score_raw_outputs.py` — local scorer with SIGALRM timeout (15 sec/call)

## Hyperparameters (current — Phase 2 plan)

| Param | Value | Rationale |
|---|---|---|
| MODEL_NAME | Qwen3-4B-Thinking-2507 | Competition mandate |
| LORA_R / ALPHA | 16 / 32 | Adequate capacity (same as exp_008) |
| LORA_DROPOUT | **0.0** | **Required** — TRL+PEFT doesn't switch eval mode reliably; dropout corrupts rollouts |
| NUM_GENERATIONS | 4 (pilot) / 8 (full) | Higher = more variance, more compute |
| MAX_PROMPT_LENGTH | 1024 | Prompt p99 ≈ 851 tokens |
| MAX_COMPLETION_LEN | 4096; consider 8192 | Phase 1B will inform — 27% of sweet-spot completions clipped at 4096 |
| LEARNING_RATE | 5e-6 | RL unstable at SFT-scale LRs |
| BETA (KL) | 0.04 | Standard; constrains policy drift from base |
| TEMPERATURE | 1.0 train / 0.6 eval | Higher exploration during training |
| TRAIN_BATCH_SIZE × ACCUM | 1 × 4 | Effective prompt batch = 4 |

### Reward signals (current)
1. **Correctness:** +1.0 if `Judger.auto_judge(post_think_text, gold, options)` passes, else 0.0
2. **Format (granular, max 0.225, additive):**
   - +0.05 if `</think>` is present
   - +0.10 if `\boxed{}` appears anywhere
   - +0.05 if `\boxed{}` appears AFTER `</think>` (proper structure)
   - +0.025 if exactly ONE `\boxed{}` post-think (no duplicates)

The granular format ensures variance across the 4 stochastic completions per prompt, so even on always-wrong prompts there's *some* gradient signal — a partial defense against the sparse-reward problem (the structural defense is curriculum filtering).

### Critical: parse after `</think>`
Qwen3-Thinking emits `<think>...</think>` then the answer. Reward extraction MUST use `extract_post_think()` — boxed answers inside the thinking block must not be rewarded, or the model learns to short-circuit reasoning.

## Dev results

_To be filled in after Phase 2 training + dev split eval._

| Metric | Baseline (exp_004) | This (dev) | Δ |
|---|---:|---:|---:|
| Overall | 55.33% | | |
| MCQ | 63.20% | | |
| Free-form | 51.40% | | |

## Topic movers

_Top 3 topics that improved / regressed — fill after Phase 2._

## Conclusion

- [ ] Keep (merge into main prompt set)
- [ ] Discard
- [ ] Needs variant — next experiment idea: ___
