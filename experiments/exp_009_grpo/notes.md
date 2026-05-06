# Experiment: grpo

**Started:** 2026-04-28
**Status (as of 2026-05-04):** Phase 1 complete, Phase 1B complete (196 sweet-spot prompts), Phase 2 (training) ready to run
**Baseline:** exp_004_fewshot_prompts (local 55.33%, Kaggle 0.551)

**Training notebook (GitHub):** https://github.com/Trevis8688/151B_SP26_Competition/blob/exp/009_grpo/experiments/exp_009_grpo/train_grpo.ipynb
**Open in Colab:** https://colab.research.google.com/github/Trevis8688/151B_SP26_Competition/blob/exp/009_grpo/experiments/exp_009_grpo/train_grpo.ipynb

## Hypothesis (unchanged)

GRPO fine-tuning on Qwen3-4B-Thinking-2507 using public.jsonl as reward signal will push accuracy beyond the prompt-engineering ceiling (~55%) by training the model to discover correct reasoning paths. Unlike SFT (which only imitates correct solutions), GRPO generates multiple responses per question and reinforces the ones that produce correct `\boxed{}` answers, allowing improvement on problems the base model currently fails.

**Key risk (materialized):** Sparse reward — with a weak base policy, most generation batches turn out uniformly wrong (or uniformly right on easy problems), producing near-zero advantage and zero gradient. The original mitigations (8 generations, format bonus) were insufficient. Solution: pre-sample to characterize per-prompt difficulty, then train only on prompts where the model gets some right and some wrong (sweet-spot curriculum filtering).

## Plan evolution

### Original plan (2026-04-28)
- One-shot GRPO via unsloth on full ~926 prompts
- 8 generations per prompt, 4096 max completion
- Train end-to-end in a single Colab session

### Current plan (2026-05-03+)
1. **Phase 1** ✅ DONE (2026-05-03): Pre-sample base model on all 926 prompts × 4 samples to characterize per-prompt difficulty (vLLM, A100, ~1.5 hr) → 147 sweet-spot prompts
2. **Phase 1B** ✅ DONE (2026-05-04): Resample 261 length-clipped prompts at `max_tokens=8192` (~1h40 A100) → 49 additional sweet-spot prompts recovered
3. **Phase 2** 🔄 READY: GRPO training filtered to **196 sweet-spot prompts** (Cell 6 of `train_grpo.ipynb` filters via `sweet_spot_ids.json` when `USE_CURRICULUM=True`)

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

## Phase 1B results (resampling at 8192 tokens, 2026-05-04)

**Setup:** Same vLLM/Qwen3 config as Phase 1, but:
- `MAX_COMPLETION_LEN=8192` (2× Phase 1)
- `VLLM_MAX_NUM_SEQS=16` (halved from 32 to fit 2× KV cache on 40GB)
- **NO judging in the notebook** — Phase 1's inline judging hung overnight on pathological LaTeX (sympy infinite loop). Phase 1B dumps raw vLLM outputs; scored locally afterward via `scripts/score_raw_outputs.py` with per-call SIGALRM timeout (15s).

**Phase 1B distribution (261 resampled prompts):**

| Bucket | Count | % | Interpretation |
|---|---:|---:|---|
| 0/4 | 193 | 73.9% | Still wrong (109 still all-clipped at 8192!) |
| 1/4 | 27 | 10.3% | Sweet spot |
| 2/4 | 10 | 3.8% | Sweet spot |
| 3/4 | 12 | 4.6% | Sweet spot |
| 4/4 | 19 | 7.3% | Easy after all — model just needed >4096 budget |
| **Sweet (1–3)** | **49** | **18.8%** | **+49 to training set** |

**Of the 261 resampled:** 109 are STILL all-clipped at 8192 — some problems genuinely need 12K+ tokens. 84 finished thinking but were just wrong. 49 became sweet-spot. 19 became always-correct (would have been fine at 4096 with more luck).

**Local scoring stats:** 1044 completions, 0 timeouts, 159 (15.2%) correct. SIGALRM safety net wasn't triggered — Phase 1's hang was a fluke that the timeout would have caught anyway.

**Final merged training set: 196 sweet-spot prompts** (up from 147 Phase 1 only)
- 80 MCQ + 116 free-form
- ~33% larger than Phase 1 alone

## Phase 2: GRPO training (ready to run)

`train_grpo.ipynb` Cell 6 now filters the dataset to `sweet_spot_ids.json` when `USE_CURRICULUM=True` (default). Cell 4 uploads `sweet_spot_ids.json` alongside the other files.

**Training math (full mode, 196 prompts):**
- Steps/epoch: 196 / (batch 1 × accum 4) = **49 grad steps**
- Time/step: ~5 min on A100 with HF backend
- **~4 hours per epoch** — fits in one Colab Pro+ session
- Plan: 1 epoch first, see if reward trends up + dev eval improves; expand to 2-3 epochs if so

**Pilot smoke test first:** `PILOT_MODE=True, PILOT_N=20` for ~30 min sanity check before committing to the full 4 hr.

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
├── difficulty_samples_long.jsonl  (gitignored, ~21MB) Phase 1B scored locally, 261 records
├── difficulty_samples_merged.jsonl (gitignored, ~31MB) Phase 1+1B merged — final source of truth
├── raw_outputs.jsonl              (gitignored, ~30MB) Phase 1 raw vLLM completions backup
└── raw_outputs_long.jsonl         (gitignored, ~21MB) Phase 1B raw vLLM completions backup
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
