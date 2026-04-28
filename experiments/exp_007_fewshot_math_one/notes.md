# Experiment: fewshot_math_one

**Date:** 2026-04-27
**Baseline compared against:** exp_004_fewshot_prompts (local: 55.33%, Kaggle: pending exp_006)

## Hypothesis

Adding a single FEWSHOT_MATH example — a series-style problem with a symbolic/fractional answer — will teach the model the expected conclusion pattern (step-by-step → single `\boxed{}`) and reduce the 17% missing_boxed rate, lifting free-form accuracy from 51.4%.

## Change from baseline

- `FEWSHOT_MATH`: add 1 example. Series-style problem (targets the weakest free-form topic at 41.2%). Answer is a symbolic fraction (not an integer) to prevent regurgitation. No fake `<think>` blocks.
- `FEWSHOT_MCQ`: unchanged from exp_004 (3 examples, proven at 63.2%).
- `SYSTEM_PROMPT_MATH` / `SYSTEM_PROMPT_MCQ`: unchanged.
- Config: `split: dev`, same vLLM params as exp_004.

**What exp_005 ruled out (do not repeat):**
- No "N boxes for N values" instruction — triggers Final Answer summary block, doubles boxes, breaks judger.
- No concrete integer answers in fewshots — Qwen3-Thinking regurgitates them verbatim under uncertainty.

## Dev results

_Fill in after running analyze.py on results.jsonl (split=dev)._

| Metric | Baseline (exp_004 full) | This (dev) | Δ |
|--------|------------------------:|-----------:|---:|
| Overall | 55.33% | | |
| MCQ | 63.20% | | |
| Free-form | 51.40% | | |

## Topic movers

_Top 3 topics that improved / regressed._

## Conclusion

- [ ] Keep (merge into `main` prompt set)
- [ ] Discard
- [ ] Needs variant — next experiment idea:
