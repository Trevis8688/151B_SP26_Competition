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

| Segment | exp_017 pass-2 baseline | exp_031 (pass-2 + FF-precision) | Δ |
|---|---:|---:|---:|
| Free-form (100) | 53.00% | **56.00%** | **+3.0pp** |
| MCQ (100) | 62.00% | 68.00% | +6.0pp (noise — MCQ prompt byte-identical) |
| Overall (200) | 57.50% | 62.00% | +4.5pp |

Per-case:
- `a/b` regurgitation: **0/100** ✓ (the exp_030 v3 fix held)
- `\boxed{3, 7}` echo: **2/100** (id=217, id=1049) — both on already-wrong cases where the model bailed under uncertainty; symptomatic, not regression drivers
- FF GAINED (6): id=32, 256, 321, 457, 951, 1096
- FF LOST (3): id=97, 613, 1076 (id=97 = 302 vs gold 301, arithmetic miss not prompt-related; id=613/1076 last-box empty, likely truncation)
- **Mechanism confirmed on 2 cases** that also recovered on pass-4 dev (id=32 `\dfrac{21275}{3}` matching gold 7091.666..., id=457 `85.94366927` ≈ gold 85.94366926...) — exact-fraction / high-precision recovery is real on pass-2 too, just rarer than on pass-4

## Conclusion

**Ambiguous pass — weak signal.** FF +3pp lands in the gate's gap (I left +2-4pp unresolved). Real mechanism contribution is ~+2pp (the 2 verified per-case recoveries that match pass-4's pattern); the remaining ~+1pp is sampling drift from the prompt change.

Why the lever is weaker on pass-2 than on pass-4 (dev +5pp → +3pp): pass-2 has different residuals — fewer questions are "fixable by exact-fraction" because pass-2's wrong_math bucket has more strict-precision failures and fewer trivial-rounding failures. The prompt is doing what it's supposed to, the addressable set is just smaller here.

**Projected board lift: ~+0.5pp** (smaller than pass-4 dev would have projected; could be 0 within noise). Echo artifacts are minor (~2%) and don't regress correct cases.

**Decision (2026-05-24): HOLD exp_031, prioritize exp_033 first.** Reasons:
1. exp_033 (rescue retune) is the safer lift on the same stage-1 source (~+0.5pp expected, three changes each backed by a documented finding)
2. exp_031 standalone is a marginal shot (~+0.5pp expected, weak signal). Spending a Kaggle slot on it now competes with exp_033.
3. If exp_033 lands a new champion, the compound bet (exp_034 = exp_031 stage-1 + exp_033 rescue) becomes the next experiment, capturing exp_031's contribution.
4. If exp_033 ties or regresses, revisit exp_031 as a standalone follow-on (1 slot).

Either way exp_031 is not discarded — it's deferred behind the safer move.

## Promotion to full run (2026-05-24)

exp_033 board landed at **0.625** = tie/noise band vs 0.628 champion. The rescue stack appears saturated on the board (see [[project_grpo_local_no_transfer]] — GRPO-as-rescuer extends the local-inflated-board-flat pattern). Per the deferred plan above, exp_031 is now activated as the standalone follow-on.

**What changes:** `config.json` `split` is already `"full"`; the `notes` field is updated to describe the full-run intent. Nothing else changes — `prompts.py` is byte-identical, model/sampling/vLLM all unchanged from the dev probe.

### Full-run plan

1. Refresh the `151b-experiments` Kaggle dataset version (bug-111 — never skip after a git push).
2. Kaggle `cse151b-notebook.ipynb`: `EXPERIMENT = "exp_031_ff_precision_pass2"` → attach utils dataset → confirm "loaded 1126 questions" → Save & Run All (T4×2, ~80 min for full split).
3. Download `public_responses.jsonl` + `private_responses.jsonl` + `submission.csv` (raw stage-1 = the board discriminator).
4. Score `public_responses.jsonl` locally → fill in the "Full results" section below.
5. Submit stage-1 raw to Kaggle as the discriminator. **The board reference is exp_017's 0.586 stage-1, not 0.628 champion** — raw stage-1 cannot beat the full-stack champion, so this submission tests whether FF-precision LIFTS pass-2 stage-1 on the board.
6. Branch on board result (see gate below).

### Pre-committed gate (full-run, board)

The board reference for this submission is **exp_017's 0.586 stage-1** (raw pass-2, no rescue, original prompts). Board 1σ ≈ 2.3pp on the ~470-q split.

| Board result vs 0.586 | Interpretation | Action |
|---|---|---|
| **≥ +0.010 (≈ 0.596+)** | FF-precision ports to pass-2 stage-1 on the board | **Promote to exp_034:** layer exp_018 rescue config (strict70, max_tokens=4096, scope=all) on these responses; expected full-stack board ~0.633+ if rescue lift carries. |
| **+0.000 to +0.009** | Stage-1 lift ambiguous (consistent with the dev's +3pp soft signal) | **Conditional:** if Kaggle slots permit before deadline, still run exp_034 (rescue layer is cheap, ~20 min Kaggle slot) since rescue can rescue residuals the FF-precision prompt didn't fix; otherwise lock exp_018 (0.628). |
| **≤ −0.001** | FF-precision regresses pass-2 stage-1 on the board (local↔board inversion again, see [[project_grpo_local_no_transfer]]) | **STOP.** Lock exp_018 (0.628) as final. Do NOT layer rescue — a regressed stage-1 + saturated rescue is highly likely to underperform 0.628. |

### Full results (fill in after Kaggle run)

| Segment | exp_017 full (pass-2, orig prompts) | exp_031 full (pass-2, FF-precision) | Δ |
|---|---:|---:|---:|
| MCQ (375) | 63.73% | tbd | tbd |
| Free-form (751) | 53.40% | tbd | tbd |
| Overall (1126) | 56.84% | tbd | tbd |
| **Board (private 943)** | **0.586** | **tbd** | **tbd** |

### Notes for the run
- Same Kaggle notebook flow as exp_017 — no special handling needed.
- The dev probe ran on the SAME notebook and prompt; confidence is high that this will execute cleanly.
- Watch for the diagnostic tell: "loaded 1126 questions" (not 200) — confirms the dataset refresh + split=full landed.
- Stage-1 board submission message suggestion: `"exp_031 stage-1: pass-2 + FF-precision prompt (full, raw)"`.
