# Experiment: pass4_stage1

**Date:** 2026-05-22
**Baselines:**
- exp_020_pass3_stage1 — pass-3 stage-1 (local 58.61%, **leaderboard stage-1-only 0.586**)
- exp_017_pass2_stage1 — pass-2 stage-1 (local 56.84%, **leaderboard stage-1-only 0.586**)

## Hypothesis

Stage-1 inference with the merged GRPO **pass-4** policy (`TrevorDuong/qwen3-4b-thinking-grpo-pass4`, from exp_022 — checkpoint-130, matched-sampler curriculum). Single variable changed from exp_020: `model_id` (pass-3 → pass-4). Prompts and sampling config byte-identical to stay in-distribution with GRPO training.

The pass-4 curriculum was re-sampled from the pass-3 policy at the training-matched per-token distribution (top_k=−1, top_p=1.0, T=1.0, N=8) at the 5120 budget with `--allow-clipped`. Hypothesis: closing the long-standing top_k=20-sample / top_k=None-train gap raises the in-band-prompt fraction → less dead-step training → a stage-1 policy that scores higher on the **held-out leaderboard**.

### Tempered prior (from the exp_022 training logs)

The visible late-epoch window (steps 124–137) showed only ~7% of steps with `correctness_reward/std > 0` — **not** an improvement over the historical ~10% dead-step rate. That argues the matched curriculum did not materially raise the in-band fraction, and the dead-step root cause is policy peakedness (BnB 4-bit), as exp_022 flagged as the alternative. So expect a likely TIE at the board. (Caveat: late-epoch easy prompts are already learned; not the full-run frac.)

## Change from baseline (exp_020)

**Single variable:** `model_id` `TrevorDuong/qwen3-4b-thinking-grpo-pass3` → `TrevorDuong/qwen3-4b-thinking-grpo-pass4`. Everything else (prompts.py, max_tokens=8192, T=0.6, top_p=0.95, top_k=20, vLLM sizing) identical.

## Plan

1. **Prereq:** merge checkpoint-130 → `qwen3-4b-thinking-grpo-pass4` via `experiments/exp_022_grpo_pass4/merge_and_push.ipynb` on Kaggle (base = pass-3, subfolder = checkpoint-130).
2. Refresh the `151b-experiments` Kaggle dataset (pull this commit).
3. `cse151b-notebook.ipynb` → set `EXPERIMENT = "exp_024_pass4_stage1"` → Save & Run All (T4×2, full split).
4. Download `public_responses.jsonl` + `private_responses.jsonl` into this dir.
5. Score local: `~/miniconda3/envs/my-virtenv/bin/python scripts/score.py experiments/exp_024_pass4_stage1/public_responses.jsonl --out experiments/exp_024_pass4_stage1/results.jsonl` (diagnostic only).
6. **DECISIVE STEP — stage-1-only leaderboard submission (no rescue):**
   ```
   ~/miniconda3/envs/my-virtenv/bin/python scripts/build_submission_from_responses.py \
     experiments/exp_024_pass4_stage1/private_responses.jsonl \
     experiments/exp_024_pass4_stage1/submission_stage1only.csv
   KAGGLE_API_TOKEN="..." ~/miniconda3/bin/kaggle competitions submit \
     cse-151-b-spring-2026-competition -f experiments/exp_024_pass4_stage1/submission_stage1only.csv \
     -m "exp_024 pass-4 stage-1 ONLY (no rescue)"
   ```

## Success / abort criteria (leaderboard, per exp_022 revision)

Judge vs the **0.586** pass-2/pass-3 stage-1 floor (split noise ~2.3pp). Local % is diagnostic-only — pass-3's +1.77pp local gain transferred 0pp to the board, so local is not a valid go/no-go.

| Pass-4 stage-1 board (vs 0.586) | Interpretation | Action |
|---|---|---|
| ≥ ~0.59 (clear of split noise) | Matched-sampler curriculum is a real board lever | Layer exp_018/exp_014 rescue (new exp), submit if ≥ 0.628 |
| ~0.586 (tie) | GRPO converged on the board regardless of local | **No pass-5.** Pivot: diagnose rescue stage (why exp_018 > exp_021 by 1.7pp), or SFT v2 |
| < ~0.58 | Board regression | Discard; investigate earlier checkpoints |

## Results

_(to be filled after merge + inference + submission)_

| | exp_017 pass-2 | exp_020 pass-3 | exp_024 pass-4 |
|---|---:|---:|---:|
| Leaderboard stage-1-only | 0.586 | 0.586 | TBD |
| Local public.jsonl | 56.84% | 58.61% | TBD |
| Local MCQ | — | 66.67% | TBD |
| Local free-form | — | 54.59% | TBD |

## Conclusion

_(to be filled)_
