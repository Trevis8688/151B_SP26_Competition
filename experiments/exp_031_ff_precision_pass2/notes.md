# Experiment: ff_precision_pass2 (DEV-ONLY probe)

**Date:** 2026-05-24
**Baselines:**
- exp_017_pass2_stage1 — pass-2 stage-1, ORIGINAL prompts. Matched comparison = 200-dev subset
  (`dev_subset_results.jsonl`): **FF 53.00%, MCQ 62.00%, Overall 57.50%**.
- exp_030 (FF-precision prompt on pass-4 dev): FF 60.00% (+5.0pp vs pass-4 dev) — the prompt is validated.
- exp_018_pass2_rescue — Kaggle **0.628**, the champion / hard floor. Its base is pass-2.

## Why this experiment (2026-05-24)

GRPO scaling is exhausted: exp_029 (pass-5 stage-1) regressed to board **0.586** (−1.4pp vs pass-4's
0.600 floor) despite +1.68pp local — a full local↔board inversion. STOP GRPO; pass-2 stays the
champion's base. The only live lever left is the **FF-precision prompt** (exp_030, dev-validated +5pp FF).

**Why pass-2, not pass-4:** the goal is beating 0.628, so the cleanest path is a single-variable change
from the champion (exp_018 = pass-2 stage-1 + exp_014 rescue). exp_025 proved pass-4 full-stack (0.621)
is *below* the pass-2 stack (0.628) — rescue is non-additive and tuned to pass-2's residuals, so switching
base would trade a +0.014 stage-1 gain for a known rescue-interaction loss. The FF prompt is format-driven
(exact fractions match the judger symbolically), not capability-driven, so the +5pp lift should port to
pass-2 — same OOD shift, both bases trained on original prompts.

## Change from baseline (exp_017)

**Single variable:** swap `SYSTEM_PROMPT_MATH` to the exp_030 FF-precision text (abstract wording: exact
reduced fraction; keep symbolic constants; ≥10 sig figs for decimal-only; NO literal `\boxed{}` placeholders).
`prompts.py` byte-identical to exp_030. MCQ prompt, few-shots, model (pass-2), sampling, vLLM sizing,
split=dev — all identical to exp_017.

## Plan (dev lifecycle)

1. Commit + push; **refresh the `151b-experiments` Kaggle dataset version** (the exp_030 stale-dataset
   lesson — bug-111).
2. Kaggle: `EXPERIMENT = "exp_031_ff_precision_pass2"` → attach utils dataset → confirm "loaded 200
   questions" → Save & Run All (T4×2, ~20 min).
3. Download `public_responses.jsonl`; score; compare FF vs the pass-2 dev baseline (53.00%).

## Success / abort gate (pre-committed, dev)

Board 1σ on the full split ≈ 2.3pp; dev n=100/segment → 1σ ≈ 5pp. exp_018 (0.628) is the hard floor.

| Dev result | Interpretation | Action |
|---|---|---|
| FF ≥ ~+4pp vs pass-2 dev (≈ ≥ 57%) AND no new echo artifact (`\boxed{3,7}` or `a/b` regurgitation) | FF-precision ports to pass-2 | **Promote:** full public+private on pass-2 + this prompt + exp_014 rescue (= exp_018 config except the prompt) → board vs 0.628 |
| FF < +2pp vs pass-2 dev, OR a new echo/format artifact | Prompt does not port to pass-2's distribution | **STOP.** Lock exp_018 (0.628) as final |

MCQ is diagnostic-only (prompt byte-identical → sampling noise; do not gate on it — see
[[feedback_per_segment_prompt_noise]]).

## Results

_(to be filled after the dev run)_

| Segment | exp_017 pass-2 baseline | exp_031 (pass-2 + FF-precision) | Δ |
|---|---:|---:|---:|
| Free-form (100) | 53.00% | TBD | TBD |
| MCQ (100) | 62.00% | TBD | TBD |
| Overall (200) | 57.50% | TBD | TBD |

Per-case: any `\boxed{3, 7}` echo leaks? __ ; any `a/b` regurgitation? __ ; exact-fraction recoveries present? __

## Conclusion

_(to be filled)_
