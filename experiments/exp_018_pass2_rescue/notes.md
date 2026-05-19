# Experiment: pass2_rescue

**Date:** 2026-05-19
**Baseline compared against:** exp_014_rescue_v2_grpo (Kaggle 0.611, local 59.50%)

## Hypothesis

exp_017 demonstrated that the GRPO pass-2 policy lifts stage-1 by +0.89pp overall (+1.20pp free-form, +0.26pp MCQ) vs exp_009 stage-1. The exp_014 rescue stack lifted exp_009 by +3.55pp (55.95% → 59.50%) primarily via MCQ rescue (+1.33pp local) with a base of +2.22pp from missing_boxed recovery.

If the +0.89pp stage-1 gain compounds with the existing rescue lift, exp_018 lands near **60.0–60.5% local**, which would beat exp_014's 0.611 on Kaggle.

Realistic expectations:
- **Best case:** lift is additive → ~60.4% local, ~0.617–0.622 Kaggle.
- **Likely case:** lift partially overlaps with rescue's gains (pass-2 already fixed some cases that the rescue would have caught) → ~60.0%, ~0.612–0.615 Kaggle.
- **Worst case:** pass-2 generates longer chains, so more responses hit `max_tokens=8192` → more missing_boxed → rescue has to work harder but at the same 4096 budget. Could be flat or slightly regress vs exp_014.

The 176 missing_boxed in exp_017 (vs 78 in exp_014's stage-1 = exp_009) suggests the **worst case is real risk.** Pass-2 may have shifted the truncation rate up. Watch this metric in the rescue results.

## Change from baseline (exp_014)

**Single variable changed:** `source_experiment` from `exp_009_grpo` to `exp_017_pass2_stage1`.

Everything else is byte-identical to exp_014:
- Rescue model: `TrevorDuong/qwen3-4b-thinking-grpo-strict70` (NOT the pass-2 model — isolating policy change to stage-1 only)
- Rescue prompts: `prompts.py` is a literal copy of exp_014's
- `max_tokens=4096`, `temperature=0.1`, `top_p=0.95`, `top_k=20`
- Same vLLM sizing (max_model_len=8192, max_num_seqs=24, tp=2)
- Same `max_input_tokens_from_stage1=3000`

## Plan

### 1. Upload exp_017 stage-1 responses as Kaggle dataset
You need both `public_responses.jsonl` and `private_responses.jsonl` from the exp_017 Kaggle run.

- `public_responses.jsonl` is already in `experiments/exp_017_pass2_stage1/`
- `private_responses.jsonl` — download from the exp_017 Kaggle notebook output (sidebar → /kaggle/working/private_responses.jsonl)

Then on Kaggle:
1. Create new dataset → name: `exp-017-pass2-stage1-responses` (must match `stage1_dataset_name` in this experiment's config.json exactly)
2. Upload both `.jsonl` files
3. Publish

### 2. Refresh 151b-experiments dataset
- Open dataset → New Version → Update from GitHub `main` → triggers re-import of `experiments/exp_018_pass2_rescue/`

### 3. Run rescue_notebook.ipynb on Kaggle
1. Open `rescue_notebook.ipynb` at repo root on Kaggle
2. Attach the **`exp-017-pass2-stage1-responses`** dataset (created in step 1)
3. Attach the **`151b-experiments`** dataset (already attached if you've been using it)
4. Cell 3: change `RESCUE_EXPERIMENT = "exp_018_pass2_rescue"`
5. Save & Run All (Commit) on T4×2 — expect ~15–20 min

### 4. Download + score locally
- Download `public_responses.jsonl`, `private_responses.jsonl`, `submission.csv`, `rescue_stats.json`
- Drop into `experiments/exp_018_pass2_rescue/`
- Score:
  ```
  ~/miniconda3/envs/my-virtenv/bin/python scripts/score.py \
    experiments/exp_018_pass2_rescue/public_responses.jsonl \
    --out experiments/exp_018_pass2_rescue/results.jsonl
  ```
- `/analyze experiments/exp_018_pass2_rescue/results.jsonl`
- `/compare exp_014_rescue_v2_grpo exp_018_pass2_rescue`

## Success / abort criteria

| Local (public.jsonl) vs exp_014 (59.50%) | Interpretation | Action |
|---|---|---|
| ≥ +0.5pp (i.e., ≥ 60.0%) | Compounded gain | Submit to Kaggle |
| 0.0 to +0.5pp | Marginal compounding | Submit if free-form improved AND missing_boxed didn't grow; otherwise skip submission |
| < 0.0pp | Pass-2 didn't compound | Discard. Implies pass-2 fixed cases the rescue was already fixing. Next move: pass-2 as RESCUER (exp_019) instead of as stage-1 |

## Results

_(to be filled after Kaggle run)_

| Metric | exp_014 baseline | exp_018 | Δ vs exp_014 |
|--------|-----------------:|--------:|-------------:|
| Local overall | 59.50% | TBD | — |
| Local MCQ | 72.53% | TBD | — |
| Local free-form | 53.00% | TBD | — |
| Missing_boxed (pre-rescue) | 78 | TBD | — |
| Missing_boxed (post-rescue) | TBD | TBD | — |
| Rescue rate (correct/candidates) | TBD | TBD | — |
| Kaggle (private) | 0.611 | TBD | — |

## Conclusion

_(to be filled)_

## Next lever

- [ ] If exp_018 wins → submit. Then evaluate exp_019: pass-2 as rescuer
- [ ] If exp_018 ties → exp_019 with pass-2 as rescuer (different lever)
- [ ] If exp_018 regresses → investigate missing_boxed growth; if pass-2's longer chains are the culprit, exp_017 with `max_tokens=12288` could fix at stage-1
- [ ] Pass-2 difficulty resample on DSMLP completes in parallel → gates GRPO pass 3 only if exp_018 confirms pass-2 is the right direction
