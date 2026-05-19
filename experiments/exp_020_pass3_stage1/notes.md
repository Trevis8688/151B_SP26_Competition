# Experiment: pass3_stage1

**Date:** 2026-05-19
**Baselines:**
- exp_017_pass2_stage1 (local 56.84%) — direct stage-1 comparison
- exp_018_pass2_rescue (Kaggle 0.628, local 60.39%) — current best stack

## Hypothesis

The pass-3 GRPO policy (trained from pass-2 on a fresh 72-prompt curriculum sampled from pass-2 itself) should produce a measurable stage-1 lift over exp_017's pass-2 stage-1 (56.84%).

Realistic expectations conditional on exp_019 training health:
- **If pass-3 ran 14 steps and recipe is consistent:** expect roughly 1/3 of exp_017's per-step gain → **+0.3pp local at stage-1** (vs exp_017's +0.89pp from 48 steps)
- **If pass-3 hit reward_std=0 anywhere:** training degraded → no lift or regression
- **Stretch:** pass-3 generalizes more efficiently → +0.5-0.8pp (matching pass-2)

## Change from baseline (exp_017)

**Single variable changed:** `model_id` from `qwen3-4b-thinking-grpo-pass2` → **`qwen3-4b-thinking-grpo-pass3`**

Everything else identical to exp_017:
- Same prompts (`prompts.py` is a literal copy of exp_017's)
- Same sampling: T=0.6, max_tokens=8192, top_p=0.95, top_k=20
- Same vLLM sizing: max_model_len=10240, max_num_seqs=32, tp=2

## Pre-requisite

The pass-3 model must be merged + pushed to HF Hub before this can run:
- Merge `TrevorDuong/qwen3-4b-thinking-grpo-pass3-ckpt/checkpoint-N` (where N is the final step from exp_019 training; will be 14 if config holds) into the pass-2 base via `experiments/exp_015_grpo_pass2/merge_and_push.ipynb` — change the constants:
  - `BASE = "TrevorDuong/qwen3-4b-thinking-grpo-pass2"`
  - `ADAPTR = "TrevorDuong/qwen3-4b-thinking-grpo-pass3-ckpt"`
  - `TARGET = "TrevorDuong/qwen3-4b-thinking-grpo-pass3"`
  - `SUBFOLDER = "checkpoint-N"` (replace N with the actual final step)
- After merge, flip the pass-3 HF repo to **public** so the Kaggle notebook can pull without HF auth (same workflow as pass-2 — saves notebook editing).

## Plan

1. exp_019 training completes (or final checkpoint pushed if pod hits 12h)
2. Run `merge_and_push.ipynb` on Kaggle with pass-3 constants
3. Verify HF Hub upload at `https://huggingface.co/TrevorDuong/qwen3-4b-thinking-grpo-pass3`; flip to public
4. Refresh `151b-experiments` Kaggle dataset (re-pull from GitHub main)
5. Open `cse151b-notebook.ipynb` on Kaggle → set `EXPERIMENT = "exp_020_pass3_stage1"`
6. Save & Run All on T4×2 — ~80 min
7. Download `public_responses.jsonl` + `private_responses.jsonl` into this folder
8. Score: `~/miniconda3/envs/my-virtenv/bin/python scripts/score.py experiments/exp_020_pass3_stage1/public_responses.jsonl --out experiments/exp_020_pass3_stage1/results.jsonl`
9. `/compare exp_017_pass2_stage1 exp_020_pass3_stage1`

## Success / abort criteria

| Local (vs exp_017 stage-1 56.84%) | Interpretation | Action |
|---|---|---|
| ≥ +0.5pp (≥ 57.34%) | Strong continuation of pass-2 trend | Layer rescue (exp_021), submit if ≥ 61.0% local |
| 0.0 to +0.5pp | Modest lift; expected given step count | Still layer rescue (exp_021), evaluate vs current 0.628 Kaggle best |
| < 0.0pp | Pass-3 regressed | Discard; try ckpt-10 (earlier checkpoint); if still regression, GRPO well is dry → SFT v2 |

## Results

_(to be filled after Kaggle run)_

| Metric | exp_017 stage-1 | exp_020 stage-1 | Δ |
|--------|----------------:|----------------:|---:|
| Local overall | 56.84% | TBD | — |
| Local MCQ | 63.73% | TBD | — |
| Local free-form | 53.40% | TBD | — |
| Missing_boxed count | 176 | TBD | — |
| Kaggle (private) | not submitted | not submitted (dev step) | — |

## Conclusion

_(to be filled)_

## Next lever

- [ ] If pass-3 ≥ exp_017: scaffold exp_021 (pass-3 + exp_014 rescue stack)
- [ ] If pass-3 < exp_017: evaluate earlier checkpoints; if all regress, pivot to SFT v2
