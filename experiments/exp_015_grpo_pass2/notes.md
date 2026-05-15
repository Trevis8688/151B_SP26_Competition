# Experiment: grpo_pass2

**Date:** 2026-05-14
**Baseline compared against:** exp_014_rescue_v2_grpo (Kaggle 0.611, local 59.50%)

## Hypothesis

Continuing GRPO training from the exp_009 checkpoint on a fresh difficulty-v2 curriculum (intermediate-difficulty prompts re-sampled from the current GRPO model) will yield +0.5–1.5pp Kaggle over exp_014 (target ≥ 0.616). The lever is fresh gradient signal — the original strict-70 curriculum was sampled from base Thinking-2507, so many prompts are now 4/4 (too easy, zero reward variance) or 0/4 (too hard) for the GRPO model and contribute nothing to learning.

## Change from baseline (exp_009 — the last GRPO training run)

| Knob | exp_009 | exp_010 Run 2 (failed) | **exp_015** | Rationale |
|---|---|---|---|---|
| Starting model | `Qwen/Qwen3-4B-Thinking-2507` (base) | base | **`TrevorDuong/qwen3-4b-thinking-grpo-strict70`** | Already +3.2pp; no reason to retrain from scratch |
| KL reference | base (TRL default = start model) | base | **exp_009 GRPO** (= start model, TRL default) | ref=start → true continuation; ref=base → partially undoes exp_009's gains |
| Curriculum | `sweet_spot_ids_clean.json` (70 prompts, from base) | same (stale) | **difficulty_v2 output** (re-sampled from GRPO model) | Fresh intermediate-difficulty prompts for the current policy |
| `max_completion_length` | 6144 | 6144 (**OOM at step 13**) | **4096** | Fixes TRL entropy_from_logits OOM; strict-70 prompts completed within 4096 in sampling |
| LR | 2e-5 | 2e-5 | 2e-5 | Hold constant — isolate curriculum + checkpoint change |
| BETA | 0.01 | 0.01 | 0.01 | Hold constant |
| num_generations | 4 | 4 | 4 | Hold constant |
| Epochs | ~0.8 (interrupted) | 1 | 1 | Hold constant |
| LoRA approach | new adapter on base | new adapter on base | **new adapter on merged exp_009 model** | exp_009 hub model is a full-weight merge (8GB safetensors), not adapter-only |

**Only three things change vs exp_009: starting checkpoint, curriculum source, max_completion_length.**

## Hard dependency

**Cannot launch until `scripts/sample_difficulty_v2.py` completes on DSMLP.** The output JSONL must be post-processed to extract intermediate-difficulty IDs (1–3/4 correct) as the new curriculum. Filter script TBD — write once the JSONL arrives.

## Curriculum composition (open question)

After difficulty_v2 finishes, inspect the MCQ/free-form split of intermediate-difficulty prompts. If most MCQ prompts are now 4/4 (trivial for the GRPO model), the curriculum will be free-form-weighted — ideal, since the 329 wrong_math cases are all free-form. If still MCQ-heavy, consider filtering to 2:1 free-form:MCQ.

## Success / abort criteria

| Kaggle | Interpretation | Action |
|---|---|---|
| ≥ 0.616 | GRPO scaling confirmed | Keep; consider pass 3 |
| 0.608–0.615 | Flat vs exp_014 | GRPO likely exhausted; pivot to SFT v2 |
| < 0.608 | Regression | Discard; pivot to SFT v2 |

## Implementation plan

1. Wait for difficulty_v2 JSONL on DSMLP
2. Filter to 1–3/4 correct IDs → `experiments/exp_015_grpo_pass2/curriculum_v2.json`
3. Write training script (adapt `exp_010/train_grpo_v2.py`):
   - `base_model = TrevorDuong/qwen3-4b-thinking-grpo-strict70`
   - `ref_model` unset (TRL default = base_model → true continuation)
   - `max_completion_length = 4096`
   - `curriculum_file = curriculum_v2.json`
4. Push to main → SSH DSMLP → launch A5000 batch pod
5. Push adapter checkpoints to HF Hub every 10 steps (survives DeadlineExceeded)
6. After training: merge adapter → push merged model → run Kaggle inference

## Results

_(to be filled in after training + Kaggle run)_

| Metric | exp_014 (baseline) | exp_015 | Δ |
|---|---:|---:|---:|
| Local overall | 59.50% | | |
| Local MCQ | 72.53% | | |
| Local free-form | 53.00% | | |
| Kaggle | 0.611 | | |

## Conclusion

- [ ] Keep
- [ ] Discard
- [ ] Needs variant
