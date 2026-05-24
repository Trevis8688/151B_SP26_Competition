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

Ran on Kaggle (dev, 200 q). Scored vs the exp_024 200-dev subset.

| Segment | exp_024 baseline | exp_027 (v1) | exp_028 (v2) | Δ v2 vs baseline |
|---|---:|---:|---:|---:|
| Free-form (100) | 55.00% | 58.00% | **60.00%** | **+5.0pp** |
| MCQ (100) | 70.00% | 66.00% | 71.00% | +1.0pp (noise) |
| Overall (200) | 62.50% | 62.00% | 65.50% | +3.0pp |

FF flips vs baseline: **net +5 (6 gained, 1 lost)** — cleaner than v1's +3.

Per-case checks:
- **Symbolic protection WORKED:** id=429 recovered (v1 False → v2 True), boxed `8, e^2, \text{growth}`. The leak is plugged.
- id=32 ✓ (`\dfrac{21275}{3}`), id=457 ✓ (`85.94366927`), id=1027 ✓ (`470x-390, \frac{456}{47}`) all stayed correct.
- **NEW artifact — id=217 regressed (v1 True → v2 False): the model boxed the literal `a/b`.** It copied the placeholder out of the prompt's `(for example \boxed{a/b})`. This is the documented exp_005 / [[feedback_no_concrete_fewshot_answers]] failure — Qwen3-Thinking echoes concrete example tokens under uncertainty.
- Only other loss: id=973 (gold `C`, an FF-classified *letter* question; model gave `30682`). Same edge case that churned in v1 — sampling, not our change.

## Conclusion

**Net win (+5pp FF) and the symbolic fix is confirmed — but the prompt has a copyable-placeholder leak.**
Putting `\boxed{a/b}` (and `\boxed{e^2}`, `\boxed{3\sqrt{2}}`) as literal examples invites regurgitation;
id=217 proved it. Fix is trivial: drop all boxed examples, phrase the instruction abstractly ("exact
reduced fraction", "keep symbolic form").

**Next: exp_030 (v3)** — same instruction, no literal `\boxed{}` examples. Verify on dev that (1) FF
holds ≥ +5pp, (2) id=217 comes back AND id=429 stays fixed, (3) no new echo artifact. If clean →
promote to the full public+private stage-1 run + stage-1-only board test, then layer exp_018 rescue.
**exp_018 (0.628) is the hard floor.**
