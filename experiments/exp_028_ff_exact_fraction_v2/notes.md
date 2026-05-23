# Experiment: ff_exact_fraction_v2 (DEV-ONLY probe)

**Date:** 2026-05-23
**Baselines:**
- exp_024_pass4_stage1 — pass-4 stage-1, original prompts. Matched comparison = the **200-dev subset**
  of its `public_responses.jsonl` (already scored: FF 55.00%, MCQ 70.00%, overall 62.50%).
- exp_027_ff_exact_fraction — v1 of this probe (FF 58.00% / +3.0pp, MCQ 66.00%, overall 62.00%).
- exp_018_pass2_rescue — Kaggle **0.628**, the hard floor.

## Hypothesis

exp_027 confirmed the precision mechanism per-case: 4 unambiguous exact-fraction / extended-precision
recoveries (id=32, 217, 457, 1027) traceable to the instruction, including id=32 which was
pre-identified in `reports/2026-05-23_rescue_saturation_pass4.md` (prediction, not narrative-fit). The
MCQ −4pp was pure sampling noise (MCQ prompt byte-identical, yet 0/100 MCQ responses identical).

But exp_027 also **introduced one regression**: id=429, where the model converted symbolic `e^2` →
`7.389056099` (gold wanted symbolic `e^2`), scored wrong. The v1 prompt covered rational→fraction and
decimal-only→sig-figs but said nothing about **closed-form symbolic constants**.

v2 plugs that leak: keep π / e / √ / surds in exact symbolic form, only decimalize when the answer can
*only* be a decimal. Expect the FF gain to hold or grow (id=429-class cases come back) with no new
failure mode.

## Change from baseline (exp_027)

**Single variable:** `SYSTEM_PROMPT_MATH` adds a clause protecting closed-form symbolic constants
(keep `\pi`, `e`, `\sqrt{}` symbolic; don't convert to decimal). Everything else — MCQ prompt,
few-shots, model (pass-4), sampling (T=0.6/top_p=0.95/top_k=20), vLLM sizing, split=dev — identical.

## Plan (dev lifecycle)

1. Commit + push; refresh the `151b-experiments` Kaggle dataset.
2. Kaggle: `EXPERIMENT = "exp_028_ff_exact_fraction_v2"` → attach utils dataset → Save & Run All (T4×2,
   200 q, ~20 min).
3. Download `public_responses.jsonl` (200 rows) into this dir; score with `scripts/score.py`.
4. Compare vs the exp_024 200-dev baseline (FF 55 / MCQ 70) AND vs exp_027.

## Success / abort gate (pre-committed, dev)

| Dev result | Interpretation | Action |
|---|---|---|
| FF ≥ +3pp vs exp_024 baseline AND id=32/217/457/1027 stay correct AND id=429 recovered, no new FF failure mode | Mechanism confirmed, leak plugged | Promote: full public + private stage-1 run → stage-1-only board test; then layer exp_018 rescue. Target board > 0.628. |
| FF gain shrinks below +3pp, OR a new symbolic/format failure mode appears | Prompt sensitivity / capability ceiling | **STOP.** Lock exp_018 (0.628) as final. |

MCQ is **diagnostic-only** here — its delta is sampling noise (prompt unchanged); do not gate on it.
**exp_018 (0.628) is the hard floor — never submit anything worse.**

## Results

_(to be filled after the dev run)_

| Segment | exp_024 baseline | exp_027 (v1) | exp_028 (v2) | Δ v2 vs baseline |
|---|---:|---:|---:|---:|
| Free-form (100) | 55.00% | 58.00% | TBD | TBD |
| MCQ (100) | 70.00% | 66.00% | TBD | TBD |
| Overall (200) | 62.50% | 62.00% | TBD | TBD |

Per-case checks: id=32/217/457/1027 still correct? __ ; id=429 recovered? __ ; new failure modes? __

## Conclusion

_(to be filled)_
