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

| | exp_017 pass-2 | exp_020 pass-3 | exp_024 pass-4 | exp_029 pass-5 |
|---|---:|---:|---:|---:|
| Leaderboard stage-1-only | 0.586 | 0.586 | 0.600 | **0.586** |
| Local public.jsonl | 56.84% | 58.61% | 60.75% | **62.43%** |
| Local MCQ | — | — | 70.13% | 73.87% |
| Local free-form | — | — | 56.06% | 56.72% |

Local stage-1 (full public, matched: original prompts, same sampling): pass-5 **+1.68pp overall**
over pass-4 (MCQ +3.74pp, FF +0.66pp) — the gain was almost entirely MCQ.

Board: **0.586 = −1.4pp vs the 0.600 pass-4 floor.** Trips the `< ~0.59` regression branch.

## Conclusion

**REGRESSION — discard pass-5. STOP GRPO.** A textbook local↔board inversion: +1.68pp local (MCQ-driven)
→ −1.4pp board. The MCQ local gain was overfitting to public.jsonl, not generalization; on the held-out
board it landed back at the pass-2/pass-3 plateau (0.586). This is the second consecutive sign (after
pass-3's +1.77pp local → 0pp board) that GRPO scaling is no longer a live board lever — pass-4's +1.4pp
was a one-time step off the plateau, not the start of a climb. See [[project_grpo_local_no_transfer]].

**pass-4 stands as the best GRPO stage-1** (board 0.600 stage-1; 0.621 full stack via exp_025). Note even
the pass-4 full stack (0.621) is below the **exp_018 champion (0.628)**, which remains the locked best.

**Decisions:**
- DISCARD pass-5; do NOT build a pass-6 curriculum. Kill/abandon the DSMLP `launch_difficulty_pass5.sh`
  sampling job — it was conditional on exp_029 ≥ ~0.610 and that gate failed.
- Redirect all remaining time (~1 wk) to the **FF-precision track** (exp_030, dev-validated +5pp FF).
  Layer it on the **pass-2** base (the 0.628 champion's base), NOT pass-4. Reason: the goal is beating
  0.628, so the cleanest path is a single-variable change from the champion config (exp_018 = pass-2
  stage-1 + exp_014 rescue). exp_025 proved pass-4 full-stack (0.621) is *below* the pass-2 stack (0.628)
  because rescue is non-additive and tuned to pass-2's residuals — switching base would trade a +0.014
  stage-1 gain for a known rescue-interaction loss. The FF prompt is format-driven, not capability-driven,
  so the +5pp dev lift should port to pass-2 (same OOD shift; both trained on original prompts).
- Plan: (1) dev-probe pass-2 + exp_030 prompt (~20min, verify FF lifts ~+5pp on pass-2 too); (2) if yes,
  full public+private on pass-2 + exp_030 prompt + exp_014 rescue = exp_018 config except the prompt →
  board vs 0.628. Check full-run output for any `\boxed{3, 7}` echo leaks before locking the prompt.
