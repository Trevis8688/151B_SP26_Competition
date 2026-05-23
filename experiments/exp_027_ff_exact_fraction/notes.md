# Experiment: ff_exact_fraction (DEV-ONLY probe)

**Date:** 2026-05-23
**Baselines:**
- exp_024_pass4_stage1 — pass-4 stage-1, original prompts (local 60.75%, board stage-1-only **0.600**). The matched comparison is the **200-dev subset** of its existing `public_responses.jsonl` (all 200 dev ids confirmed present — no new baseline run needed).
- exp_018_pass2_rescue — Kaggle **0.628**, the hard floor. This probe never submits anything worse.

## Hypothesis

The `wrong_math` diagnostic (see `reports/2026-05-23_rescue_saturation_pass4.md`) found a
**precision bucket** in pass-4's free-form errors: ~50 cases on the full public set where the
model had the right values but they were judged wrong, because the judger's tolerance is
`precision=1e-08` (near-exact) and the model rounded to 2–4 digits. Proven recoverable: re-boxing
the full-precision value (or an exact fraction — SymPy matches `3100/7` against decimal gold
symbolically) flips these to correct.

The chain-precision discriminator showed only ~12 of 50 already carry full precision in the
reasoning chain (safely re-extractable). The other ~36 only ever computed the rounded value, and
most of those golds are repeating decimals = **clean fractions**. A 4B model emits an exact
fraction far more reliably than 12 decimal digits, so a **stage-1 free-form prompt instruction**
("rational → exact fraction; else ≥10 sig figs; never round") could recover a slice of the full 50,
not just the 12.

**This is a question-independent model-behavior fix** (the model rounds the same way on public and
private), so unlike GRPO local gains it should transfer to the private board near 1:1. But it carries
the exp_005 risk: prompt changes can knock Qwen3-Thinking out of distribution and regress MCQ. Hence
a cheap dev probe BEFORE spending any board slot.

## Change from baseline (exp_024)

**Single variable:** `SYSTEM_PROMPT_MATH` (free-form only) gains the exact-fraction / no-round
instruction. `SYSTEM_PROMPT_MCQ`, the 3 MCQ few-shots, `model_id` (pass-4), sampling
(T=0.6/top_p=0.95/top_k=20), and vLLM sizing are byte-identical to exp_024. `split` set to `dev`
(notebook runs only on `data/splits/dev.jsonl`, skips private inference + submission).

## Plan (dev lifecycle)

1. Commit + push this folder to main; refresh the `151b-experiments` Kaggle dataset.
2. On Kaggle: `cse151b-notebook.ipynb` → `EXPERIMENT = "exp_027_ff_exact_fraction"` → attach utils
   dataset (judger) → Save & Run All (T4×2). Dev = 200 q, ~20 min, no private run.
3. Download `public_responses.jsonl` (200 rows) into this dir.
4. Score: `~/miniconda3/envs/my-virtenv/bin/python scripts/score.py experiments/exp_027_ff_exact_fraction/public_responses.jsonl --out experiments/exp_027_ff_exact_fraction/results.jsonl`.
5. Build the **matched baseline** by scoring the 200-dev subset of exp_024's existing responses
   (filter `experiments/exp_024_pass4_stage1/public_responses.jsonl` to the 200 dev ids, score),
   then compare free-form accuracy (this exp vs baseline) AND MCQ accuracy (regression watch).

## Success / abort gate (pre-committed, dev)

100 FF + 100 MCQ in dev → 1σ ≈ 5pp per segment, so require a clear move, not noise.

| Dev result | Interpretation | Action |
|---|---|---|
| FF ↑ materially (≥ +3pp) AND MCQ flat (within ±2pp) | Precision fix works, no OOD regression | Full public + private run; build stage-1-only board submission; layer rescue only if it beats 0.628 |
| FF flat/down OR MCQ regresses (≥ −3pp) | Format/capability ceiling, or exp_005 OOD hit | **STOP.** Lock exp_018 (0.628) as final. Do not spend a board slot. |

**exp_018 (0.628) is the hard floor throughout — never submit anything worse.**

## Results

Ran on Kaggle (dev, 200 q). Scored vs the 200-dev subset of exp_024's responses.

| Segment | exp_024 baseline (200-dev subset) | exp_027 (this) | Δ |
|---|---:|---:|---:|
| Free-form (100) | 55.00% | 58.00% | **+3.0pp** |
| MCQ (100) | 70.00% | 66.00% | −4.0pp |
| Overall (200) | 62.50% | 62.00% | −0.5pp |

**The MCQ −4pp is sampling noise, NOT a regression.** The MCQ system prompt + few-shots are
byte-identical between the two runs, yet **0/100 MCQ responses are byte-identical** (T=0.6 stochastic,
different batch composition → different RNG). MCQ churn was 6 gained / 10 lost — balanced noise. The
pre-committed gate's MCQ guard was designed to catch exp_005-style OOD regression *from a prompt
change*; since the MCQ prompt didn't change, the guard cannot measure what it was built for. Override
is principled.

**The FF +3pp is real mechanism, not net noise.** Of 7 FF gains, **4 are unambiguous exact-fraction /
extended-precision recoveries** directly attributable to the instruction:
- id=32: `7091.67` → `\dfrac{21275}{3}` (gold 7091.666…) — **pre-identified in the report as a precision case** (prediction, not narrative-fit)
- id=217: `2.55, 4.42` → `51/20, 4.4167295593` (gold 2.55, 4.41672955930064)
- id=457: `85.94` → `85.94366927` (gold 85.9436692696235)
- id=1027: `…9.70` → `\frac{456}{47}` (gold …9.70212765957447)

The 4 FF losses are mostly sampling churn (id=147/256/973 — value/format divergence), **except id=429,
which exposes a new failure mode the instruction introduced:** the model converted symbolic `e^2` →
`7.389056099` (gold wanted symbolic `e^2`), scored wrong. The prompt covers rational→fraction and
decimal-only→sig-figs but says nothing about **closed-form symbolic constants** (π, e, √, surds).

## Conclusion

**Gate: PASS-WITH-REFINEMENT, do not STOP.** FF mechanism confirmed per-case (not just net); MCQ guard
mis-fired on sampling noise (identical prompt). But the prompt has a symbolic-constant leak (id=429).

Sizing (advisor): 4 recoveries/100 FF → ~30/751 FF → ~+2.7pp full-public stage-1; after ~70% transfer
discount ≈ +1.9pp board ≈ board 1σ (2.3pp). Positive-EV but ~coin-flip on one board attempt — worth
de-noising the prompt first rather than spending a slot on a confounded version.

**Next:** exp_028 — refine the instruction to **keep closed-form symbolic constants symbolic** (protect
the id=429 class), re-run dev once (~20 min, no board slot). Verify (1) FF gain holds/grows, (2) the 4
recovered ids stay recovered, (3) no new failure mode. If confirmed → full public + private stage-1 run,
then layer the exp_018 rescue (stage-1 fix is FF-targeted, rescue is MCQ-weighted → should be more
additive than pass-4's stage-1 was). **exp_018 (0.628) is the hard floor; submit only if board > 0.628.**
