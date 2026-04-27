# Experiment: baseline_replay

**Date:** 2026-04-26
**Baseline compared against:** exp_001_longer_context (Kaggle: 0.526)

## Hypothesis
exp_004's prompts were validated locally at **55.33% on the full public set** (judger), but the run only scored public — `dev.jsonl` wasn't on Kaggle so it fell back to public, and `split=dev` skipped private inference, so no submission.csv was ever produced. exp_005 then attempted an aggressive multi-value format-discipline change and **regressed to 42.98%** locally (Kaggle: 0.42), confirming that the exp_004 prompt was already near a stable local optimum and that prompt-level format constraints can backfire by triggering the model's "Final Answer summary" pattern (gold of length N gets answered with 2N boxes → judger length mismatch).

This experiment ships exp_004's exact prompts and config to Kaggle as a clean full-split submission. Expected Kaggle ≈ 0.55, materially beating both exp_001 (0.526) and exp_005 (0.42).

## Change from baseline
- Prompts: byte-identical to exp_004 (`SYSTEM_PROMPT_MATH`, `SYSTEM_PROMPT_MCQ`, `FEWSHOT_MCQ` × 3, `FEWSHOT_MATH` empty).
- Config: byte-identical to exp_004 except `notes` text (already had `split: full` after the earlier fix).
- Notebook: same logic, only `EXPERIMENT` string updated.
- No max_tokens bump, no temperature change, no new few-shot. Pure replay.

Why no improvements layered in: exp_005 demonstrated that even a single targeted change can collapse multi-value accuracy by 32pp. With no compute budget for dev splits, the priority is to bank the validated baseline first; iterate from a known-good Kaggle submission next.

## Dev results
_Skipped — running full inference directly. Fill in this table from Kaggle results after submission._

| Metric | exp_004 (local judger) | This (Kaggle) | Δ |
|--------|----------------------:|--------------:|---:|
| Overall (Kaggle public) | 55.33% | | |
| MCQ | 63.20% | | |
| Free-form | 51.40% | | |

## Topic movers
_N/A — replay; topic mix should match exp_004 exactly._

## Conclusion
- [ ] Keep — Kaggle ≥ 0.526
- [ ] Discard — Kaggle < 0.526 (would imply unexpected variance vs exp_004 local)
- [ ] Needs variant — next experiment idea: layer max_tokens bump or single-example FEWSHOT_MATH on top of validated baseline
