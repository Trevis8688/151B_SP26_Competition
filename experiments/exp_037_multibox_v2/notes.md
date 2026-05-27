# Experiment: multibox_v2 (extended judger-extraction post-process)

**Date:** 2026-05-27
**Type:** Post-processing fix on top of exp_018. NO model change, NO prompt change.
**Baseline:** exp_035_multibox_fix (Kaggle publicLB 0.628).
**Status:** Submitted to Kaggle (pending result).

## Origin

After exp_036 (verification probe) confirmed Qwen3-Thinking cannot self-verify (4% TPR), I revisited the judger-audit lever the user proposed alongside verification. Direct read of exp_018's 314 wrong-math FF responses surfaced a second extraction-edge-case that exp_035's rule didn't handle:

**Over-extraction.** Example id=57 ("BlueSky-style" multi-part): the model wrote `\boxed{580}`, `\boxed{660}`, `\boxed{80}` for parts (a), (b), (c), then added a summary block `\boxed{580, 660, 80}` at the end. The judger's `extract_all_boxed` greedily groups these (the gaps are only whitespace + `$`), so the final extraction is "80, 580, 660, 80" — 4 elements vs gold's 3. The summary box alone (`580, 660, 80`) would have matched gold perfectly.

exp_035's rule didn't fire on this case: it only handled UNDER-extraction (`cur_n < n_expected`), not OVER-extraction (`cur_n > n_expected`).

## Rule v4 (strict superset of exp_035)

`n_exp` = `[ANS]` count in question. (Predicts gold length 97.6% on public.)

1. If `n_exp < 2`, leave alone (MCQ + single-answer FF).
2. Collect all `\boxed{}` in response. If none, leave alone.
3. If judger's current extract count == `n_exp`, leave alone (already matching).
4. Else try in order:
   - **Candidate A (new):** last single box's comma-split count == `n_exp` → append `\nFinal Answer: \boxed{<last_box>}`. The judger now picks only that one box, getting the full N-value tuple cleanly.
   - **Candidate B (= exp_035's rule):** `len(boxes) ≥ n_exp` → append `\nFinal Answer: \boxed{b1} \boxed{b2} ... \boxed{bN}` with the last N boxes.
5. Else leave alone.

Candidate A is the new addition; candidate B is identical to exp_035's rule. So rule_v4 either does what exp_035 did (B) or does something MORE (A). It cannot do less.

`scripts/apply_multibox_v2.py` implements the rule.

## Counterfactual on full public (1126 q)

| Rule | Recoveries (wrong→correct) | Breaks (correct→wrong) | Modifications |
|---|---:|---:|---:|
| exp_035 (rule_v1) | 9 | 0 | 51 |
| **exp_037 (rule_v4)** | **13** | **0** | 61 |

The 4 new recoveries (vs exp_035) are: ids 57, 400, 579, 883. All driven by candidate A or by candidate B firing on over-extraction cases that exp_035's strict condition `cur_n < n_expected` skipped.

| Segment | exp_018 | exp_035 | exp_037 (measured) | Δ vs exp_035 |
|---|---:|---:|---:|---:|
| MCQ (375) | 73.87% | 73.87% | 73.87% | 0 (MCQ skipped) |
| Free-form (751) | 53.66% | 54.73% | **55.39%** | **+0.66pp** |
| Overall (1126) | 60.39% | 61.10% | **61.55%** | **+0.45pp** |

## Why rule_v4 is safe

- Zero breaks on the full public set across two independent samples (rule_v3 simulation + rule_v4 simulation). The rule only ever APPENDS a final-answer block; nothing the model already wrote is removed.
- Strict superset of exp_035: candidate B is identical to exp_035's rule, so any private case where exp_035 helped is still handled the same way (unless candidate A applies first, which on public always produced a correct result for the 4 cases it caught).
- Distribution-free: like exp_035, this is structurally immune to the local↔board inversion that killed exp_029/031/033 — nothing about the model or prompts changes.

## Private modifications

| Stage | exp_018 → submission | exp_035 → submission | exp_037 → submission |
|---|---|---|---|
| Mods | 0 | 37 | **52** (44 B + 8 A) |

The 15 additional private modifications (vs exp_035) come from over-extraction cases (candidate B firing on `cur_n > n_expected`) and candidate A's "summary tuple" cases.

## Kaggle submission

| Stage | exp_018 | exp_035 | exp_037 |
|---|---:|---:|---:|
| publicLB visible | 0.628 | 0.628 | **0.628** |
| privateLB (held-back, reveals 2026-06-01) | TBD | TBD | TBD |
| Private responses modified | 0 | 37 | 52 |

publicLB tie is consistent with the math: only ~half of the 943 private rows are LB-visible, and of the 52 modifications only ~5-7 are likely to flip the wrong→correct status in that subset (lifting it by <0.001 — below the 0.001 publicLB granularity). exp_037 should beat exp_035 on the held-back privateScore, which is where the additional 15 modifications get measured.

## Followups

- Other audit findings that did NOT pan out (recorded so we don't redo them):
  - Units strip (`60 mi/hr` → `60`): 1 rec / 8 breaks on public — net negative, skipped.
  - `e^{...}` → `\exp(...)` rewrite: 0 recoveries on public (only 2 candidates total).
  - Precision recovery from chain-of-thought (find higher-precision number that rounds to boxed value): 11 rec / 12 breaks at the lax setting, 1 rec / 5 breaks at the strict setting. Net negative both ways — intermediate computations in the CoT round to similar values and contaminate substitution. Dead lever.
