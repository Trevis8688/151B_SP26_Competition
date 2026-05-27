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

**Probe eval engine + budget (advisor-corrected):** the gate uses HF `generate`
at **batch=8, max_new_tokens=4096** — NOT batch=16/8192. KV-cache math: Qwen3-4B
fp16 ≈ 147KB/token, so 16×8192 ≈ 19GB KV + 8GB weights ≈ 27GB would OOM the 24GB
A5000. batch=8 @ ~5k tokens ≈ 6GB KV + 8GB = ~14GB, safe. 4096 gen tokens is
ample for a forgetting/format gate (answers reach `\boxed{}` well before then).
Probe eval ≈ ~1h at this setting.

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

## Pre-launch verification (do NOT skip — ~15 min saves a ~3h bad bet)

The launch script bakes in two fast fail-fast checks (TRL SFT API import +
SFTConfig kwargs, run in the sanity block before any training). Residual risks
to verify with a tiny dry run before trusting the full pipeline:

1. **TRL 0.21 completion-only masking.** `DataCollatorForCompletionOnlyLM` +
   `formatting_func` should mask the prompt and compute loss only on the
   assistant turn (`<|im_start|>assistant\n` response template). Verify once:
   print the first batch's `labels` and confirm prompt tokens are `-100` and only
   the assistant content (including `</think>`) carries loss. If TRL 0.21 ignores
   `max_seq_length` under `formatting_func`, truncation may not apply — check the
   token lengths in the first batch.
2. **HF-generate OOM.** Even at batch=8/4096 do a 10-q dry run of `eval_dev.py`
   first; confirm no CUDA OOM before the real 200-q gate.
3. **MathQA format mismatch (advisor note).** MathQA is 5-option lowercase
   `a)..e)` with sometimes-thin `Rationale`; competition MCQ is up to 10 options
   `A..J`. `prepare_data.py` maps to uppercase `A..E` + `Options:` block, which is
   close but not identical. If MCQ regresses at the probe gate, look here first —
   the SFT MCQ signal is gradient (not a memorized template) so it should
   generalize, but the style gap is the most likely culprit.

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

## TRL 0.21.0 SFT API (verified against v0.21.0 source, 2026-05-24)

First launch died at the sanity check (the fail-fast worked as designed) on
`ImportError: cannot import name 'DataCollatorForCompletionOnlyLM' from 'trl'`.
That collator was removed from TRL's top-level export by 0.21. Verified the
correct API against `trl/trainer/sft_config.py` + `sft_trainer.py` @ v0.21.0:

- **No `DataCollatorForCompletionOnlyLM`, no `formatting_func`.** SFTTrainer
  handles dataset formats natively in `_prepare_dataset`.
- **Field is `max_length`, not `max_seq_length`** (renamed; the old name errors).
- **Use conversational prompt-completion format:** each row = `prompt` (list of
  system+user message dicts) + `completion` (list with the assistant dict). When
  a `prompt` column is present, `completion_only_loss` auto-enables and the prompt
  is masked by *prefix length* (line ~774) — no `{% generation %}` chat-template
  tags needed, so this is robust regardless of Qwen3's template internals.
- Container resolved: `trl=0.21.0 peft=0.19.1 transformers=4.57.6 torch=2.5.1`.

Data format changed accordingly: `prepare_data.py` now emits
`{"prompt": [system,user], "completion": [assistant]}` (was `{"messages": [...]}`)
and prepends the exp_017 system prompt so SFT examples match the inference prompt
format (few-shots omitted per-example to keep sequences short).

## Results (fill in after each phase)

### Phase A — probe (recovered via re-eval pod trduong-3314570 on 2026-05-26)

The original launch pod was deleted before its gate output could be read.
`scripts/reeval_sft_v2_probe.sh` pulled the probe adapter from
`TrevorDuong/qwen3-4b-pass2-sft-v2-probe/checkpoint-final` and re-ran
`eval_dev.py` on the 200q dev split.

| Signal | Threshold | Probe result | Pass? |
|---|---:|---:|:-:|
| MCQ dev % | 60.0 | **23.00%** (23/100) | **FAIL** (−37.0pp) |
| FF dev % | 53.0 | **33.00%** (33/100) | **FAIL** (−20.0pp) |
| `\boxed{}` extraction % | 95.0 | **68.50%** (137/200) | **FAIL** (−26.5pp) |

**Verdict: catastrophic forgetting + format degradation.** MCQ collapsed −39pp vs the pass-2 dev baseline (62.00%) — worse than exp_008's −22pp collapse. The two fixes that supposedly avoided exp_008's mistake (Thinking-family base + MCQ data mix via AQuA-RAT) were not enough.

The 31.5% missing-\boxed{} rate is the most surprising signal: every SFT training example had `\\boxed{{answer}}` at the end of the completion, and the prompt-completion completion-only-loss masking *should* have preserved that format adherence. Three plausible root causes (NOT investigating further given the time budget):
1. Inference-time chat template mismatch — the probe adapter saved its own `chat_template.jinja`; the tokenizer at inference time may have applied a different one, breaking the prompt → completion boundary the model learned.
2. The `<think>...</think>` wrapper around the rationale in training examples may have driven the model to spend its 4096 token budget inside `<think>` and never emit `\\boxed{}`.
3. AQuA-RAT's narrative MCQ rationale style ("The answer is E.") may have taught a free-form output mode that overrides Thinking-2507's `\boxed{}` reflex.

### Phase B and beyond — not run

Phase A FAIL is unconditional STOP per the pre-committed decision tree. Phase B (full train) and Phase C (Kaggle inference + board) were not executed.

### Phase C — full

Not run. Phase A FAIL aborted the pipeline.

### Stage-1 board

Not run. Phase A FAIL aborted the pipeline.

## Final conclusion (2026-05-26)

exp_034 SFT v2 is dead. The probe gate caught catastrophic forgetting in ~3h instead of letting an 8h full train + Kaggle inference + dev eval cycle waste ~12h confirming it. The gate design worked as intended.

All three remaining-lever candidates have now run their course:
- GRPO scaling: dead (exp_029 board 0.586, regression vs pass-4)
- Rescue retuning: dead (exp_033 board 0.625, tie/regression vs 0.628)
- Prompt-tuning on GRPO base: dead (exp_031 board 0.568, regression)
- SFT on GRPO base: dead (exp_034 probe gate −39pp MCQ)

**exp_018 (Kaggle 0.628, local 60.39%) is the final competition score.**

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
