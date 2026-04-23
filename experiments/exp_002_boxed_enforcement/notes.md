# Experiment: boxed_enforcement

**Date:** 2026-04-22
**Baseline compared against:** exp_001_longer_context (Kaggle: 0.526)

## Hypothesis
Strengthening the system prompt to explicitly require ending with `\boxed{}` will reduce the 22% missing_boxed rate (258/1126 responses had no `\boxed{}` anywhere, costing points even on Kaggle's lenient judger). Few-shot deferred to exp_003 to avoid interfering with Qwen3 thinking model's internal CoT.

## Change from baseline
- `SYSTEM_PROMPT_MATH`: added "you MUST end your response with your final answer in \boxed{}" and "Do not stop before writing \boxed{}"
- `SYSTEM_PROMPT_MCQ`: added same enforcement + "Do not write anything after \boxed{}"
- No few-shot examples (kept empty) — isolate prompt-format effect first

## Results

| Metric | Baseline (exp_001) | This | Δ |
|--------|---------:|-----:|---:|
| Overall (local) | 3.14% | 3.05% | -0.09% |
| Kaggle | 0.526 | 0.491 | -0.035 |
| Has `\boxed{}` | 868/1126 (77%) | 764/1126 (68%) | -9% |

## Conclusion
- [ ] Keep (merge into `main` prompt set)
- [x] Discard
- [ ] Needs variant — next experiment idea:

Enforcement wording made things worse — missing_boxed increased from 23% to 32%. The stricter instruction disrupted the thinking model's CoT flow, causing more responses to never finish. Revert to exp_001 prompts as baseline.
