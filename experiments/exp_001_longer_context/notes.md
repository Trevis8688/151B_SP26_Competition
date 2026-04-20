# Experiment: longer_context

**Date:** 2026-04-20
**Baseline compared against:** exp_000_starter_baseline (Kaggle: 0.48)

## Hypothesis
Increasing max_model_len from 6144 to 16384 will allow the thinking model to finish its chain-of-thought and write \boxed{}, fixing the ~97% truncation rate observed in exp_000.

## Change from baseline
- `max_model_len`: 6144 → 16384
- `max_num_seqs`: reduced to avoid OOM (lower concurrency is acceptable)
- `max_num_batched_tokens`: adjusted accordingly

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
