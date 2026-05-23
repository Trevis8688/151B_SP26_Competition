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

_(to be filled after the dev run)_

| Segment | exp_024 baseline (200-dev subset) | exp_027 (this) | Δ |
|---|---:|---:|---:|
| Free-form (100) | TBD | TBD | TBD |
| MCQ (100) | TBD | TBD | TBD |
| Overall (200) | TBD | TBD | TBD |

## Conclusion

_(to be filled)_
