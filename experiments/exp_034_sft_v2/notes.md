# Experiment: sft_v2 (Hail Mary, DSMLP A5000)

**Date:** 2026-05-24
**Type:** Training experiment — DSMLP A5000, QLoRA SFT on `qwen3-4b-thinking-grpo-pass2`.
**Status:** scaffolded; not yet launched.
**Baseline (champion):** exp_018 Kaggle **0.628**, local 60.39%.
**Stage-1 floor for board comparison:** exp_017 Kaggle **0.586** (pass-2 raw stage-1).

## Why now (2026-05-24)

The "if GRPO stalls" lever (documented in CLAUDE.md). GRPO has demonstrably stalled:
- pass-5 (exp_029) board: 0.586, −0.014 vs pass-4's 0.600 floor. Full local↔board inversion.
- exp_033 (rescue retune): 0.625, −0.003 vs champion. Rescue stack saturated.

All scaling-flavored levers exhausted. SFT v2 is the only honest remaining swing — fresh mechanism, not just rescaling the same dial. ~6 days left in competition; DSMLP idle.

## Fixes vs exp_008's two root causes

| exp_008 root cause | exp_034 fix |
|---|---|
| Base was `Qwen3-4B` (non-thinking) → MCQ collapsed −22pp | Base is `TrevorDuong/qwen3-4b-thinking-grpo-pass2` (Thinking family, with GRPO already baked in) |
| Trained on NuminaMath-CoT free-form only → catastrophic MCQ forgetting | Mix in MathQA (2k MCQ examples) alongside NuminaMath (5k FF). Wrap both in `<think>...</think>\n\n\boxed{{answer}}` |

## Hypothesis

SFT on Thinking-base + balanced MCQ/FF mix produces a better-calibrated stage-1 model than the GRPO-only pipeline plateaus at (pass-2 = 0.586, pass-4 = 0.600, pass-5 = 0.586). Even a modest stage-1 lift (≥+0.010 board over 0.586) can carry through the saturated rescue stack to land ≥0.633 final — beating the 0.628 champion.

## Plan (refined per advisor)

Three load-bearing fixes vs my initial strawman:

### 1. Two-phase train with hard dev-probe gate at hour ~3

The biggest single risk-reducer. Prevents an exp_008-style multi-day loss.

```
Phase A: probe (~1h)
  - Train on 500 mixed examples (357 FF + 143 MCQ, same 5:2 ratio as full)
  - 1 epoch, save adapter
Phase B: dev gate (~10 min)
  - vLLM inference on data/splits/dev.jsonl (200q) with probe adapter
  - Compute MCQ %, FF %, boxed{} extraction rate
  - GATE (pre-committed):
      MCQ ≥ 60%   AND   FF ≥ 53%   AND   extraction ≥ 95%
  - If ANY fails: abort, push diagnostic logs to HF, DO NOT continue
Phase C: full (~6-8h, conditional)
  - Resume from probe adapter, train remaining 6500 examples
  - HFPushAdapterCallback every 50 steps -> resume across 12h pod boundary
```

### 2. Data spec — decided in plan, not deferred

| Field | Value | Notes |
|---|---|---|
| FF source | `AI-MO/NuminaMath-CoT` | 5000 random examples |
| MCQ source | `allenai/math_qa` | 2000 random examples — well-known math MCQ benchmark |
| Format | `<think>{rationale}</think>\n\n\\boxed{{answer}}` | Both sources wrapped programmatically |
| MCQ answer encoding | letter (a..e) | from MathQA's `correct` option index |
| Max sequence length | 2048 | examples over length are dropped |
| Seed | 42 | reproducible split |

The wrapper format is the load-bearing detail: it preserves the Thinking architecture's `<think>` head behavior, which exp_008 corrupted by training on plain CoT without thinking tags.

### 3. HFPushAdapterCallback for 12h pod survival

DSMLP container ceiling is 12h. QLoRA on 7k examples at batch 1×16 ≈ 6-8h, but the probe + full + overhead margin is tight. Crib `HFPushAdapterCallback` + `_try_resume_from_hf()` from `experiments/exp_026_grpo_pass5/train_grpo.py` lines 358-453 — pushes adapter + optim + scheduler state every `save_steps`, and the relaunch script picks up where it left off.

## Change from baseline (exp_017 pass-2 stage-1)

Single variable: **the model weights**. `prompts.py` is **not** in this experiment dir because SFT changes the model itself; inference will use the standard `cse151b-notebook.ipynb` with the new model id pointed at the SFT-merged HF repo, and the original (exp_017) prompts.

## Pre-committed gates

### Gate A — Probe gate (~3h in)

Computed from `eval_dev.py` on `data/splits/dev.jsonl` (200q, 100 MCQ + 100 FF) using the probe adapter.

| Signal | Threshold | Source / motivation |
|---|---:|---|
| MCQ dev % | ≥ **60.0** | pass-2 dev MCQ = 62.00% (exp_017 dev_subset). >5pp drop = forgetting started — exp_008 path |
| FF dev % | ≥ **53.0** | pass-2 dev FF = 53.00% (exp_017 dev_subset). Below baseline = SFT hurting target |
| `\boxed{}` extraction % | ≥ **95.0** | GRPO baked format adherence; SFT can wash it out silently |

**Failure action:** abort, log adapter to HF as `*-probe-failed`, halt the launcher, do NOT spend 5-7 more hours.

### Gate B — Final gate (after full train)

| Stage | Threshold | Action |
|---|---|---|
| Dev (200q) | MCQ ≥ 60% AND FF ≥ 55% | Promote to stage-1 board test (one Kaggle slot) |
| Stage-1 board (vs 0.586 floor) | ≥ +0.010 (~0.596+) | Promote to rescue layer (exp_035 = SFT stage-1 + exp_018 rescue) |
| Stage-1 board | < +0.010 | STOP. Lock exp_018 (0.628) as final. SFT did not transfer. |

### Gate C — Time-budget abort

If probe + full + dev + stage-1 board isn't done by **2026-05-27 (day 3)**, abandon SFT v2 and lock exp_018. Leaves 3 days for safe-mode reverts and any final sanity-check submission.

## Files in this experiment dir

| File | Purpose |
|---|---|
| `config.json` | All hyperparameters + gate thresholds |
| `prepare_data.py` | Download NuminaMath + MathQA → wrap → save train.jsonl + probe.jsonl |
| `train_sft.py` | QLoRA trainer with HF-Hub-checkpoint callback; `--phase probe\|full` |
| `eval_dev.py` | vLLM inference on dev split, returns per-segment % + extraction rate |
| `requirements.txt` | Python deps for the DSMLP venv |
| `notes.md` | This file |

Launch script: `scripts/launch_sft_v2.sh` (sibling, not in this dir).

## Launch sequence

From `dsmlp-login`:

```bash
# 0. Ensure HF token is on the home PVC
[ -f ~/.hf_token ] || echo "MISSING ~/.hf_token" && exit 1

# 1. Push experiment dir to main (mandatory — DSMLP git pull picks up from main)
cd ~/151B_SP26_Competition && git fetch origin main && git reset --hard FETCH_HEAD

# 2. Launch (single command, batch mode; entire probe→gate→full pipeline runs detached)
bash scripts/launch_sft_v2.sh

# 3. Monitor
kubectl get pods
kubectl logs -f <pod_name>

# 4. If the 12h cap hits mid-full, relaunch — _try_resume_from_hf picks up from latest pushed step
bash scripts/launch_sft_v2.sh
```

## Results (fill in after each phase)

### Phase A — probe (after ~1h)

| Signal | Threshold | Probe result | Pass? |
|---|---:|---:|:-:|
| MCQ dev % | 60.0 | tbd | tbd |
| FF dev % | 53.0 | tbd | tbd |
| `\boxed{}` extraction % | 95.0 | tbd | tbd |

### Phase C — full (after ~6-8h more)

| Segment | exp_017 dev baseline | SFT-v2 full dev | Δ |
|---|---:|---:|---:|
| MCQ (100) | 62.00% | tbd | tbd |
| FF (100) | 53.00% | tbd | tbd |
| Overall (200) | 57.50% | tbd | tbd |

### Stage-1 board (after Kaggle full-set inference)

| Floor (exp_017 stage-1) | SFT-v2 stage-1 | Δ |
|---:|---:|---:|
| 0.586 | tbd | tbd |

## Risks (open list, monitor across phases)

1. **SFT overwrites GRPO gains.** Mitigated by low rank (r=16), low LR (2e-5), one epoch. Probe gate catches this as MCQ regression.
2. **Format drift.** `<think>` wrapper + boxed{} extraction can degrade. Caught by extraction-rate gate.
3. **NuminaMath / MathQA quality.** Some examples have noisy rationales. Format wrapping isolates content from structure but won't fix bad CoT.
4. **12h pod cap.** Mitigated by `HFPushAdapterCallback`.
5. **Local→board inversion.** Documented pattern for GRPO-flavored gains ([[project_grpo_local_no_transfer]]). SFT is a different mechanism, but the public.jsonl overlap caveat still applies. Stage-1 board test is the only honest discriminator.

## Decision tree (after this experiment)

```
Phase A probe fails             -> abort, lock exp_018 as final
Phase C full < gate B           -> abort, lock exp_018 as final
Stage-1 board < +0.010 vs 0.586 -> lock exp_018 as final
Stage-1 board >= +0.010         -> exp_035 = SFT stage-1 + exp_018 rescue stack
  exp_035 board < 0.633          -> lock exp_018 (champion holds within 1σ)
  exp_035 board >= 0.633         -> new champion
```
