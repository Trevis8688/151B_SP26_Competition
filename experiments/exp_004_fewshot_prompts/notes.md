# Experiment: fewshot_prompts

**Date:** 2026-04-25
**Baseline compared against:** exp_001_longer_context (Kaggle: 0.526)

## Hypothesis
Adding 2–3 few-shot MCQ examples (question + options + brief justification + `\boxed{LETTER}`, no fake `<think>` traces) will improve MCQ accuracy by giving the model a clear format template, without disrupting the thinking model's internal CoT — unlike exp_002's enforcement wording which raised missing_boxed from 23% to 32%.

## Change from baseline
- `FEWSHOT_MCQ`: 2–3 short examples drawn from public.jsonl (~200–300 tokens each). Show question + options + one-line reasoning + `\boxed{LETTER}`. No `<think>` traces — let the model do its own thinking.
- `SYSTEM_PROMPT_MCQ`: revert to exp_001 wording (no enforcement language from exp_002).
- `SYSTEM_PROMPT_MATH` + `FEWSHOT_MATH`: unchanged from exp_001 — isolate MCQ effect only.
- Config: `split: dev`, `num_samples: 1`, restore exp_001 throughput params (`max_num_seqs: 32`, `max_num_batched_tokens: 20480`, `gpu_memory_utilization: 0.90`). Should run ~1 hr on dev.

## Dev results
Note: `dev.jsonl` was not found on Kaggle (not uploaded to the dataset), so the notebook fell back to the full public set (1126 questions). `split=dev` was still set, so private inference was skipped — no submission.csv. Full-run re-submitted with `split=full` to get Kaggle score.

Scores below use the Kaggle judger (`Judger.auto_judge()`), as run inside the notebook. Baseline (exp_001) Kaggle score = **0.526**; its local-judger breakdown is not recorded here since the notebooks used different scoring paths.

| Metric | Baseline (exp_001 Kaggle) | This (public set, judger) | Δ |
|--------|--------------------------:|-------------------------:|---:|
| Overall | ~52.6% | **55.33%** (623/1126) | +2.7 pp |
| MCQ | — | **63.20%** (237/375) | — |
| Free-form | — | **51.40%** (386/751) | — |

## Topic movers
_Run `scripts/analyze.py experiments/exp_004_fewshot_prompts/results.jsonl` once full-run results land._

## Conclusion
- [x] Keep — 55.33% overall on public set (judger), MCQ up to 63.2%. Awaiting Kaggle score from full-split re-run (split=full pushed to main 2026-04-26).
- [ ] Discard
- [ ] Needs variant — next experiment idea:
