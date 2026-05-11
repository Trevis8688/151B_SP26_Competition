# Experiment: best_of_n

**Date:** 2026-05-11
**Baseline:** exp_009_grpo (local 55.95%, Kaggle **0.583**)

## Hypothesis

Best-of-N sampling with N=3 at T=0.6 on top of the exp_009 GRPO model attacks two failure modes simultaneously:

1. **Missing-boxed (16.3% of errors).** The notebook's `majority_vote` filters to samples that have a properly-extracted `\boxed{}` answer. If 1 of 3 samples produces a proper box where the others ran out of tokens, we capture it. P(at least one boxed) = 1 - (1 - 0.84)^3 ≈ 99.6% vs 84% at N=1.

2. **Sampling noise on borderline problems.** For problems near the model's accuracy boundary, majority voting over 3 stochastic rollouts smooths out variance — flips some 50/50 questions into the correct bucket.

## Change from baseline (exp_009)

| Knob | exp_009 | exp_011 | Rationale |
|---|---|---|---|
| model_id | TrevorDuong/qwen3-4b-thinking-grpo-strict70 | (same) | Best model we have |
| num_samples | 1 | **3** | The whole experiment |
| temperature | 0.6 | 0.6 | Keep — same as training eval temp |
| max_tokens | 8192 | 8192 | Keep — generous budget |
| vllm.max_num_seqs | 32 | **12** | KV cache is 3× larger per slot; throttle to fit |

The active Kaggle notebook (`cse151b-notebook.ipynb`) already wires `NUM_SAMPLES → vllm.SamplingParams(n=...)` and applies `majority_vote()` post-inference — no notebook changes needed.

### How `majority_vote` works (in the current notebook)

```python
def majority_vote(texts):
    # extract \boxed{...} from each of N samples
    answers = [(extract_boxed(t), t) for t in texts]
    valid   = [(ans, t) for ans, t in answers if ans is not None]
    if not valid:
        return texts[0]                          # all-missing → fallback
    modal = Counter(ans for ans, _ in valid).most_common(1)[0][0]
    return next(t for ans, t in valid if ans == modal)
```

- MCQ: picks the letter that appears most often. ✓
- Free-form: picks the boxed value that appears most often. ✓
- All-missing: still returns texts[0]. So we don't *create* new boxes — we just *prefer* samples that have one. The rescue benefit is N×: even one boxed sample wins over zero.

## Runtime estimate

exp_009 (single-sample) ran inference for ~3 hours on Kaggle T4 x2 over 1126+943 = 2069 questions. At N=3, generation work is ~3× that = ~9 hours, perilously close to Kaggle's 9hr session cap.

**Mitigations:**
- `max_num_seqs=12` (down from 32) keeps memory in budget for 3 concurrent samples per prompt
- If the run threatens to time out, kill and drop to `num_samples=2` (~6 hr expected) — still solves the missing-boxed rescue (P(at least one boxed) at N=2 is ~97% vs 84%)
- Consider splitting: one Kaggle session for public, one for private (if a single 9hr session is too tight)

## Expected gain

- **Optimistic:** if 1/3 boxed samples rescues all the missing_boxed: ~+5pp local. Probably overshoots — boxed-success isn't independent across samples.
- **Realistic:** rescue ~50% of the 16.3% missing-boxed failures + small majority-vote MCQ gain ≈ **+1.5 to +3pp local**.
- **Pessimistic:** if the model's errors are systematic (not sampling noise) and the missing-boxed cases share a common cause (too-long CoTs), N=3 barely helps. Floor: 0pp.

## Dev results

_Run on Kaggle (full split) — fill in after results.jsonl downloaded._

| Metric | Baseline (exp_009) | This | Δ |
|---|---:|---:|---:|
| Overall | 55.95% | | |
| MCQ | 63.47% | | |
| Free-form | 52.20% | | |
| missing_boxed errors | 183 (16.3%) | | |

## How to run

On Kaggle:
1. Refresh the `151b-experiments` dataset (it auto-pulls from `main` branch — must merge exp/011 first)
2. In `cse151b-notebook.ipynb` Cell 3, set `EXPERIMENT = "exp_011_best_of_n"`
3. Save Version → Save & Run All (Commit)
4. Download `submission.csv` + `public_responses.jsonl` from the output

Local scoring after download:
```bash
~/miniconda3/envs/my-virtenv/bin/python scripts/score.py \
    experiments/exp_011_best_of_n/public_responses.jsonl \
    --out experiments/exp_011_best_of_n/results.jsonl
~/miniconda3/envs/my-virtenv/bin/python scripts/analyze.py \
    experiments/exp_011_best_of_n/results.jsonl
```

## Conclusion
- [ ] Keep
- [ ] Discard
- [ ] Needs variant — next experiment idea:
