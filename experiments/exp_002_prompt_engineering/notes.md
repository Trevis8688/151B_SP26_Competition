# Experiment: prompt_engineering

**Date:** 2026-04-22
**Baseline compared against:** exp_001_longer_context (Kaggle: 0.526)

## Hypothesis
Strengthening the system prompt to explicitly require ending with `\boxed{}` will reduce the 22% missing_boxed rate (258/1126 responses had no `\boxed{}` anywhere, costing points even on Kaggle's lenient judger). Few-shot deferred to exp_003 to avoid interfering with Qwen3 thinking model's internal CoT.

## Change from baseline
- `SYSTEM_PROMPT_MATH`: added "you MUST end your response with your final answer in \boxed{}" and "Do not stop before writing \boxed{}"
- `SYSTEM_PROMPT_MCQ`: added same enforcement + "Do not write anything after \boxed{}"
- No few-shot examples (kept empty) — isolate prompt-format effect first

## Dev results
_Fill in after running analyze.py on results.jsonl (split=dev)._

| Metric | Baseline | This | Δ |
|--------|---------:|-----:|---:|
| Overall | | | |
| MCQ | | | |
| Free-form | | | |

## Topic movers
_Top 3 topics that improved / regressed._

## Conclusion
- [ ] Keep (merge into `main` prompt set)
- [ ] Discard
- [ ] Needs variant — next experiment idea:
