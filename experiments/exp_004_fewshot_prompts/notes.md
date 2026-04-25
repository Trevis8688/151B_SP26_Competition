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
