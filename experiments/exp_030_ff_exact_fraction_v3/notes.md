# Experiment: ff_exact_fraction_v3 (DEV-ONLY probe)

**Date:** 2026-05-23
**Baselines:**
- exp_024_pass4_stage1 — pass-4 stage-1, original prompts. Matched comparison = 200-dev subset
  (FF 55.00%, MCQ 70.00%, overall 62.50%).
- exp_027 (v1, exact-fraction): FF 58.00% — found the symbolic leak (id=429).
- exp_028 (v2, +symbolic protection): FF **60.00% (+5.0pp)**, id=429 recovered, BUT id=217 regressed by
  echoing the literal `\boxed{a/b}` placeholder.
- exp_018_pass2_rescue — Kaggle **0.628**, the hard floor.

## Hypothesis

The FF-precision instruction is a real lever (FF 55 → 58 → 60 across v1→v2). The only remaining defect in
v2 is self-inflicted: literal `\boxed{}` examples in the instruction get regurgitated as final answers
(id=217 boxed `a/b`), the documented exp_005 / no-concrete-fewshot-answers failure. Removing all boxed
examples and keeping the instruction abstract should recover id=217 while preserving the +5pp FF gain and
the id=429 symbolic fix — yielding a clean prompt to promote to the board.

## Change from baseline (exp_028)

**Single variable:** drop the literal `\boxed{a/b}`, `\boxed{e^2}`, `\boxed{3\sqrt{2}}` examples from
SYSTEM_PROMPT_MATH; phrase abstractly ("exact reduced fraction", "keep symbolic form"). The original
`\boxed{3, 7}` multi-value example is KEPT (part of the exp_024/GRPO training prompt — removing it would
be a second variable). MCQ prompt, few-shots, model (pass-4), sampling, vLLM sizing, split=dev — identical.

## Plan (dev lifecycle)

1. Commit + push; refresh the `151b-experiments` Kaggle dataset.
2. Kaggle: `EXPERIMENT = "exp_030_ff_exact_fraction_v3"` → attach utils dataset → Save & Run All (T4×2,
   200 q, ~20 min).
3. Download `public_responses.jsonl`; score with `scripts/score.py`; compare vs exp_024 baseline + v2.

## Success / abort gate (pre-committed, dev)

| Dev result | Interpretation | Action |
|---|---|---|
| FF ≥ +5pp vs baseline AND id=217 recovered AND id=429 stays fixed AND no new echo artifact | Clean precision prompt | **Promote:** full public + private stage-1 run → stage-1-only board test; then layer exp_018 rescue. Target board > 0.628. |
| FF drops below +3pp, OR a new echo/format artifact appears, OR id=429 re-breaks | Prompt too sensitive / abstract wording too weak | **STOP.** Lock exp_018 (0.628) as final. |

MCQ is diagnostic-only (sampling noise; prompt unchanged). **exp_018 (0.628) is the hard floor.**

## Results

_(to be filled after the dev run)_

| Segment | exp_024 baseline | exp_028 (v2) | exp_030 (v3) | Δ v3 vs baseline |
|---|---:|---:|---:|---:|
| Free-form (100) | 55.00% | 60.00% | TBD | TBD |
| MCQ (100) | 70.00% | 71.00% | TBD | TBD |
| Overall (200) | 62.50% | 65.50% | TBD | TBD |

Per-case: id=217 recovered? __ ; id=429 still fixed? __ ; id=32/457/1027 still correct? __ ; new echo artifacts? __

## Conclusion

_(to be filled)_
