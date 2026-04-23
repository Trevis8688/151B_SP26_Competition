# Experiment: self_consistency

**Date:** 2026-04-22
**Baseline compared against:** exp_001_longer_context (Kaggle: 0.526)

## Hypothesis
Running 5 samples per question and taking the modal `\boxed{}` answer (self-consistency / majority voting) will reduce answer variance and improve accuracy, since the dominant failure bucket is wrong_math (51%) not missing_boxed — more samples per question should amplify the signal when the model knows the right answer but is noisy.

## Change from baseline
- `SamplingParams(n=5, temperature=0.7)` — 5 samples per question, slightly higher temperature for diversity
- Post-processing: extract `\boxed{}` from each of the 5 samples, take the mode; fall back to sample 0 if no majority
- Prompts: reverted to exp_001 (no enforcement wording — exp_002 proved that hurts)
- `split: dev` — validate on 200q first before committing to a full run
- `max_tokens`: 8192 (corrected from template's 16384)

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
