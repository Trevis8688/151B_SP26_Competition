# Experiment: pass3_rescue

**Date:** 2026-05-20
**Baseline compared against:** exp_018_pass2_rescue (Kaggle **0.628**, local **60.39%**) — current best stack
**Upstream stage-1:** exp_020_pass3_stage1 (local 58.61%; MCQ 66.67%, FF 54.59%; missing_boxed 156)

## Hypothesis

exp_020 demonstrated that the GRPO **pass-3** policy lifts stage-1 by **+1.77pp** overall (MCQ +2.94pp, FF +1.19pp) vs exp_017 pass-2 stage-1, with 20 fewer truncations (156 vs 176 missing_boxed). exp_018 then showed that a stage-1 lift **passes cleanly through** the exp_014 rescue layer (+0.89pp stage-1 → +0.89pp final, neither absorbed nor amplified).

If pass-3's +1.77pp compounds the same way, exp_021 lands near **62% local**, well above exp_018's 60.39%, and — if exp_018's ~1.9× Kaggle translation rate holds even partway — comfortably past the current 0.628 Kaggle best.

Realistic expectations:
- **Best case (clean pass-through, as exp_018):** ~62.1% local.
- **Likely case (partial overlap — pass-3 already fixed some cases the rescue would catch; exp_020 already has fewer missing_boxed, so a smaller rescue candidate pool):** ~61.0–61.7% local.
- **Worst case:** the higher stage-1 base leaves mostly capability-limited (not truncation) cases for the rescue, so rescue adds little on top of the already-strong 58.61% — still ~60.5–61.0%, i.e. ≥ exp_018.

The 156 missing_boxed (down from exp_017's 176) means a **smaller rescue candidate pool** than exp_018 had — so the rescue's absolute contribution may be slightly smaller, but it starts from a higher floor. Watch the rescued-count and post-rescue missing_boxed metrics.

## Change from baseline (exp_018)

**Single variable changed:** `source_experiment` from `exp_017_pass2_stage1` → `exp_020_pass3_stage1` (and the matching `stage1_dataset_name`).

Everything else is byte-identical to exp_018 / exp_014:
- Rescue model: `TrevorDuong/qwen3-4b-thinking-grpo-strict70` (exp_009 GRPO model — NOT pass-3; keeps the rescuer constant so this isolates the stage-1 policy change)
- Rescue prompts: `prompts.py` is a literal copy of exp_018's
- `max_tokens=4096`, `temperature=0.1`, `top_p=0.95`, `top_k=20`, `max_input_tokens_from_stage1=3000`
- Same vLLM sizing (max_model_len=8192, max_num_seqs=24, tp=2)

## Plan

### 1. Upload exp_020 stage-1 responses as a Kaggle dataset
Need BOTH `public_responses.jsonl` and `private_responses.jsonl` from the exp_020 Kaggle run.
- `public_responses.jsonl` is already in `experiments/exp_020_pass3_stage1/`.
- `private_responses.jsonl` — download from the exp_020 Kaggle notebook output (`/kaggle/working/private_responses.jsonl`) if not already saved locally.

Then on Kaggle:
1. Create new dataset → name **`exp-020-pass3-stage1-responses`** (must match `stage1_dataset_name` in this config.json exactly).
2. Upload both `.jsonl` files.
3. Publish.

### 2. Refresh 151b-experiments dataset
Open dataset → New Version → update from GitHub `main` → re-imports `experiments/exp_021_pass3_rescue/`.

### 3. Run rescue_notebook.ipynb on Kaggle
1. Open `rescue_notebook.ipynb` at repo root on Kaggle.
2. Attach **`exp-020-pass3-stage1-responses`** (step 1) + **`151b-experiments`** datasets.
3. Cell 3: set `RESCUE_EXPERIMENT = "exp_021_pass3_rescue"`.
4. Save & Run All (Commit) on T4×2 — ~15–20 min.

### 4. Download + score locally
Download `public_responses.jsonl`, `private_responses.jsonl`, `submission.csv`, `rescue_stats.json` into `experiments/exp_021_pass3_rescue/`, then:
```
~/miniconda3/envs/my-virtenv/bin/python scripts/score.py \
  experiments/exp_021_pass3_rescue/public_responses.jsonl \
  --out experiments/exp_021_pass3_rescue/results.jsonl
```
Then `/compare exp_018_pass2_rescue exp_021_pass3_rescue`.

## Success / abort criteria

| Local (public.jsonl) vs exp_018 (60.39%) | Interpretation | Action |
|---|---|---|
| ≥ +0.5pp (≥ 60.9%) | Pass-3 lift compounded through rescue | Submit — likely new best |
| 0.0 to +0.5pp | Partial overlap with rescue gains | Submit if FF improved AND post-rescue missing_boxed didn't grow; else evaluate before burning a slot |
| < 0.0pp | Pass-3 fixed cases the rescue already caught (gain absorbed) | Don't submit; pass-3 stage-1 is still the win — proceed to pass-4 |

## Results

_(to be filled after Kaggle run)_

| Metric | exp_018 baseline | exp_021 | Δ vs exp_018 |
|--------|-----------------:|--------:|-------------:|
| Local overall | 60.39% | TBD | — |
| Local MCQ | 73.87% | TBD | — |
| Local free-form | 53.66% | TBD | — |
| Public rescue candidates | 181 | TBD | — |
| Public rescued | 113 | TBD | — |
| Kaggle (private) | 0.628 | TBD | — |

## Conclusion

_(to be filled)_

## Next lever

- [ ] If exp_021 wins → submit (new best). Then pass-4 (gate already cleared by exp_020): verify `GRPOConfig().top_k` default on DSMLP + 10-step pilot for largest (G, max_completion) under the exp_010 OOM line.
- [ ] If exp_021 ties/absorbs → pass-3 stage-1 is still the win; go straight to pass-4.
