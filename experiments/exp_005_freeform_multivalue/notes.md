# Experiment: freeform_multivalue

**Date:** 2026-04-26
**Baseline compared against:** exp_004_fewshot_prompts (Kaggle: pending; local-judger 55.33% on full public)

## Hypothesis
Free-form's failure is dominated by **format discipline on multi-value answers**, not reasoning. 414 / 751 free-form questions (55%) have multi-value gold; accuracy on them is **44.2%** vs **60.2%** on single-value — a 16pp gap. 53% of free-form errors have multiple `\boxed{}` calls and 18% of all responses contain at least one empty `\boxed{}` (intermediate scratchwork polluting extraction). A 3-example `FEWSHOT_MATH` pack that demonstrates (a) one final `\boxed{}` per gold value with the correct count, (b) no `\boxed{}` for intermediate work, (c) always finishing with boxed values after `</think>`, should recover most of the multi-value gap.

The judger joins all post-`</think>` boxed values with `", "` then splits by comma, so either `\boxed{a}\boxed{b}\boxed{c}` or a single `\boxed{a, b, c}` box parses to the same list — both formats work, but the count must match gold length and empty boxes must be avoided.

## Change from baseline
- Add 3 examples to `FEWSHOT_MATH` (currently empty in exp_004):
  1. **Multi-value (3 numbers)** — e.g. quartiles question. Brief reasoning, then three separate `\boxed{}` calls in order at the end. Shows "one box per value."
  2. **Single value, symbolic** — e.g. integral with fractional answer. One final `\boxed{\frac{...}{...}}`, no intermediate boxes.
  3. **Multi-value with mixed types (formula + number)** — e.g. linear regression: equation + R². Two separate `\boxed{}` calls, demonstrating that even mixed types each get their own box.
- Each example explicitly avoids `\boxed{}` for scratchwork; only the final answer line uses it.
- Keep `FEWSHOT_MCQ` and `SYSTEM_PROMPT_MCQ` from exp_004 (gained ~+2.7pp; don't regress).
- Keep `SYSTEM_PROMPT_MATH` from exp_004 but add one explicit line: "If the question asks for N values, produce N separate `\boxed{}` calls in order at the end. Never put `\boxed{}` around intermediate work."
- Config: `split: full`, `num_samples: 1`, same throughput params as exp_004 (`max_num_seqs: 32`, `max_model_len: 10240`, `gpu_memory_utilization: 0.90`). User runs full inference directly (no dev split — compute budget).

## Dev results
_Skipped — running full inference directly. Fill in this table from Kaggle results after submission._

| Metric | Baseline (exp_004) | This | Δ |
|--------|-------------------:|-----:|---:|
| Overall (judger, public) | 55.33% | | |
| MCQ | 63.20% | | |
| Free-form | 51.40% | | |
| Free-form multi-value | 44.2% | | |
| Free-form single-value | 60.2% | | |

## Topic movers
_Top 3 topics that improved / regressed._

## Conclusion
- [ ] Keep (merge into `main` prompt set)
- [ ] Discard
- [ ] Needs variant — next experiment idea:
