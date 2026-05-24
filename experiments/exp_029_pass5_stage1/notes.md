# Experiment: pass5_stage1 (stage-1-only board test)

**Date:** 2026-05-23
**Baselines (stage-1-only board, raw, no rescue):**
- exp_017 pass-2 stage-1 → board **0.586**, local 56.84%
- exp_020 pass-3 stage-1 → board **0.586**, local 58.61% (pass-3 = 0pp board transfer)
- exp_024 pass-4 stage-1 → board **0.600**, local 60.75% ← **the floor to beat** (first GRPO board gain, matched-sampler curriculum)

## Hypothesis

Pass-4 was the first pass to move the held-out board (0.586 → 0.600), driven by the **matched-sampler
curriculum** recipe, not "another pass" per se. The open question: did that recipe **unlock ongoing
learning** (pass-5 keeps climbing) or give a **one-time step** off the converged plateau (pass-5 ties)?
Pass-5 (exp_026) trained 130 grad updates from the pass-4 base on a curriculum re-sampled from the
pass-4 policy with the same matched recipe. Its stage-1-only board score vs 0.600 is the cleanest
second data point on whether GRPO is still a live lever.

**Tempered prior:** the dead-step root cause (4-bit policy peakedness) is unaddressed, and pass-4's
+1.4pp was only ~0.6σ on the ~470-q split. A tie is a real possibility. Local % is diagnostic-only —
pass-3's +1.77pp local transferred 0pp to the board.

## Change from baseline (exp_024)

**Single variable:** `model_id` pass-4 → **pass-5** (`TrevorDuong/qwen3-4b-thinking-grpo-pass5`, the
inline-merged 8.04 GB model). Prompts (original; matches GRPO training), sampling, vLLM sizing,
split=full — all identical to exp_024.

## Plan

1. Commit + push; refresh the `151b-experiments` Kaggle dataset.
2. Kaggle: `EXPERIMENT = "exp_029_pass5_stage1"` → attach utils dataset → Save & Run All (T4×2, full
   public+private, ~80 min). Pass-5 loads directly from HF — no merge step.
3. Download `public_responses.jsonl` + `private_responses.jsonl` into this dir.
4. Score local (diagnostic): `scripts/score.py public_responses.jsonl --out results.jsonl`.
5. Build the **stage-1-only** submission and submit RAW (no rescue):
   `scripts/build_submission_from_responses.py private_responses.jsonl submission_stage1only.csv`.

## Success / abort gate (leaderboard — pre-committed)

Judge vs pass-4's **0.600** stage-1-only floor. Board 1σ ≈ 2.3pp on the ~470-q split. Local is diagnostic-only.

| Pass-5 stage-1 board (vs 0.600) | Interpretation | Action |
|---|---|---|
| ≥ ~0.610 (≥ +1pp) | Matched-sampler recipe still climbing — GRPO is a live lever | Layer rescue (clone exp_025); consider pass-6 if time |
| ~0.600 (tie) | Matched-sampler was a one-time step; GRPO converged on the board | **STOP GRPO.** pass-4 stands as best stage-1. Redirect remaining time to the FF precision track (exp_028) + rescue |
| < ~0.59 | Board regression (pass-5 hurt) | Discard; keep pass-4. Inspect earlier pass-5 checkpoints |

**exp_018 (0.628) full stack remains the champion/floor — pass-5 stage-1 raw is not expected to beat it; this run is purely to measure whether GRPO is still moving.**

## Results

_(to be filled after the stage-1-only board submission)_

| | exp_017 pass-2 | exp_020 pass-3 | exp_024 pass-4 | exp_029 pass-5 |
|---|---:|---:|---:|---:|
| Leaderboard stage-1-only | 0.586 | 0.586 | 0.600 | TBD |
| Local public.jsonl | 56.84% | 58.61% | 60.75% | TBD |

## Conclusion

_(to be filled)_
