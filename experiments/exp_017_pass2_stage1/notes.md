# Experiment: pass2_stage1

**Date:** 2026-05-18
**Baseline compared against:** exp_014_rescue_v2_grpo (Kaggle 0.611, local 59.50%)

## Hypothesis

The GRPO pass-2 policy — trained for 48 steps on a fresh 58-prompt curriculum re-sampled from the exp_009 policy itself, with a continuous `length_bonus` (max 0.05) added to the reward to break exp_009's discrete-reward collapse — has internalized more signal than pass-1. Used as the **stage-1 model** with no other changes from exp_014, this should produce a net Kaggle Δ in **+0.005 to +0.020** range over exp_014's 0.611.

Realistic expectations:
- exp_015's 58-prompt curriculum was filtered from prompts where exp_009 itself scored 1/4 — these are at the policy's *current decision boundary*. Useful signal but small (~48 grad updates).
- The length_bonus mostly served to keep `reward_std > 0`; downstream gains depend on whether the reward gradient pushed the policy toward correct answers (primary) or just toward shorter outputs (secondary).
- Most likely outcome: small lift on free-form (which is where exp_014's rescue showed it's saturated), MCQ flat-or-slight (already near ceiling at 76.8%).

## Change from baseline

Only the stage-1 model. Everything else is identical to exp_014's stage-1 step:
- `model_id`: `TrevorDuong/qwen3-4b-thinking-grpo-strict70` → **`TrevorDuong/qwen3-4b-thinking-grpo-pass2`**
- Same prompts (3 MCQ few-shots, no math few-shots) — required to stay in-distribution with GRPO training
- Same vLLM/sampling config (T=0.6, max_tokens=8192, max_model_len=10240, max_num_seqs=32)

**No rescue layer in this run.** This isolates the policy delta. If pass-2 stage-1 ≥ exp_009 stage-1, then we layer the exp_014 rescue on top as exp_018.

## Plan

1. Refresh `151b-experiments` dataset on Kaggle (commit + push triggers GitHub re-import)
2. Open `cse151b-notebook.ipynb` on Kaggle, set `EXPERIMENT = "exp_017_pass2_stage1"`
3. Save & Run All (Commit) on T4×2 — expect ~80 min
4. Download `public_responses.jsonl` + `private_responses.jsonl` + `submission.csv` into this folder
5. Score locally: `~/miniconda3/envs/my-virtenv/bin/python scripts/score.py experiments/exp_017_pass2_stage1/public_responses.jsonl --out experiments/exp_017_pass2_stage1/results.jsonl`
6. `/analyze experiments/exp_017_pass2_stage1/results.jsonl`
7. `/compare exp_009_grpo exp_017_pass2_stage1` and `/compare exp_014_rescue_v2_grpo exp_017_pass2_stage1`

## Success / abort criteria

| Local (public) vs exp_009 stage-1 (55.95%) | Interpretation | Action |
|---|---|---|
| ≥ +0.5pp | Pass-2 policy beats pass-1 stage-1 | Layer rescue (exp_018) on top, then submit |
| -0.5 to +0.5pp | Flat — pass-2 didn't move the policy | Try earlier checkpoint (ckpt-30, ckpt-20) or different rescue strategy |
| < -0.5pp | Pass-2 regressed | Discard; investigate length_bonus side-effects (too-short outputs?) |

Submission decision deferred to after the rescue layer (exp_018). Don't burn a Kaggle submission slot on bare stage-1 — it's a dev step.

## Results

_(to be filled after Kaggle run)_

| Metric | exp_009 stage-1 | exp_014 (stage-1 + rescue) | exp_017 stage-1 | Δ vs exp_009 stage-1 |
|--------|----------------:|---------------------------:|----------------:|---------------------:|
| Local (public.jsonl) | 55.95% | 59.50% | TBD | — |
| MCQ accuracy | TBD | 76.80% | TBD | — |
| Free-form accuracy | TBD | 50.87% | TBD | — |
| Missing_boxed count | TBD | 78 | TBD | — |
| Kaggle (private) | 0.583 | 0.611 | TBD (not submitted) | — |

## Conclusion

_(to be filled)_

## Next lever

- [ ] If pass-2 wins stage-1: scaffold exp_018 = exp_017 stage-1 + exp_014 rescue
- [ ] If pass-2 ties or regresses: evaluate ckpt-10/20/30/40 to find sweet spot before length_bonus over-shortened
