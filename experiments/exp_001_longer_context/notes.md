# Experiment: longer_context

**Date:** 2026-04-20
**Baseline compared against:** exp_000_starter_baseline (Kaggle: 0.48)

## Hypothesis
Increasing max_model_len from 6144 to 10240 will allow the thinking model to finish its chain-of-thought and write \boxed{}, fixing the ~97% truncation rate observed in exp_000. 10240 covers p99 input length (851 tokens) + full 8192 output budget = 9043 tokens needed.

## Change from baseline
- `max_model_len`: 6144 → 10240 (derived from p99 input 851 + max_tokens 8192)
- `max_num_seqs`: 64 → 32 (~2× slower, not 4× — acceptable trade-off)
- `max_num_batched_tokens`: 12288 → 20480
- `max_tokens`: 8192 (unchanged; was mistakenly 16384 in initial config draft)

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
