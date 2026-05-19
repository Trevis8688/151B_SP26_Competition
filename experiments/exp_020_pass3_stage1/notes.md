# Experiment: pass3_stage1

**Date:** 2026-05-19
**Baselines:**
- exp_017_pass2_stage1 (local 56.84%) — direct stage-1 comparison
- exp_018_pass2_rescue (Kaggle 0.628, local 60.39%) — current best stack

## Hypothesis (UPDATED 2026-05-19 after Path B pivot)

The pass-3 GRPO policy (trained from pass-2 on a **STRICT 88-prompt curriculum** — see exp_019 notes for Path B recipe change) should produce a measurable stage-1 lift over exp_017's pass-2 stage-1 (56.84%).

### Upstream change in pass-3 recipe (Path B)

The initial pass-3 run with loose curriculum (72 prompts, allow_clipped) was killed after 6 iterations once we observed that 50% of steps had variance collapse (3 of 6 had reward_std < 0.02). The restart uses:
- **Strict curriculum:** 88 prompts (29 MCQ / 59 FF) at 1-3 correct AND no_clip — eliminates the all-clipped disaster mode AND the all-correct trivial mode
- **max_completion_length:** 4096 → 5120 — adds headroom for prompts pass-2 solved in 4500-5000 tokens
- Everything else byte-identical to exp_015 pass-2 recipe (length_bonus, lr, beta, lora config)

### Realistic expectations conditional on exp_019 training health

| Outcome | P | E[stage-1 lift over exp_017] |
|---|---:|---:|
| Strict curriculum captures the available gradient: 12+ effective grad updates | 0.50 | +0.30 to +0.50pp |
| Mixed: ~8 effective grad updates (partial collapse on the 3-correct band) | 0.30 | +0.10 to +0.25pp |
| Poor: residual variance collapse on most steps; KL accumulates without learning | 0.15 | -0.10 to +0.05pp |
| Disaster: OOM at 5120 mid-training (~17% memory margin from exp_010's OOM point) | 0.05 | n/a — fall back to ckpt-10 |

**Headline E[Δ]: +0.28pp local at stage-1** (range: -0.10 to +0.50pp)
**Stretch:** pass-3 generalizes more efficiently because strict band includes 2-3 correct prompts → +0.5 to +0.8pp (matching pass-2's +0.89pp lift)

## Change from baseline (exp_017)

**Single variable changed:** `model_id` from `qwen3-4b-thinking-grpo-pass2` → **`qwen3-4b-thinking-grpo-pass3`**

Everything else identical to exp_017:
- Same prompts (`prompts.py` is a literal copy of exp_017's)
- Same sampling: T=0.6, max_tokens=8192, top_p=0.95, top_k=20
- Same vLLM sizing: max_model_len=10240, max_num_seqs=32, tp=2

## Pre-requisite

The pass-3 model must be merged + pushed to HF Hub before this can run:
- Merge `TrevorDuong/qwen3-4b-thinking-grpo-pass3-ckpt/checkpoint-N` (where N is the final step from the Path B exp_019 restart; expected ~17-18 grad updates from 88-prompt curriculum) into the pass-2 base via `experiments/exp_015_grpo_pass2/merge_and_push.ipynb` — change the constants:
  - `BASE = "TrevorDuong/qwen3-4b-thinking-grpo-pass2"`
  - `ADAPTR = "TrevorDuong/qwen3-4b-thinking-grpo-pass3-ckpt"`
  - `TARGET = "TrevorDuong/qwen3-4b-thinking-grpo-pass3"`
  - `SUBFOLDER = "checkpoint-N"` (replace N with the actual final step — likely 17 or 18, not 14)
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
| ≥ +0.5pp (≥ 57.34%) | Path B strict curriculum confirmed | Layer rescue (exp_021), submit if ≥ 61.0% local; pass-4 likely worth pursuing |
| +0.2 to +0.5pp | Modest lift consistent with E[Δ] | Layer rescue (exp_021), evaluate vs current 0.628 Kaggle best; pass-4 marginal |
| 0.0 to +0.2pp | Below E[Δ] — strict curriculum didn't fix enough | Skip rescue submit; pivot to SFT v2 |
| < 0.0pp | Pass-3 regressed despite Path B | Discard; try ckpt-10 (earlier checkpoint); GRPO well confirmed dry → SFT v2 |

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
