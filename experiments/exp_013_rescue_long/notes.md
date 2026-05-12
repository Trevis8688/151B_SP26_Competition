# Experiment: rescue_long

**Date:** 2026-05-12
**Baseline:** exp_012_boxed_rescue (local 57.99%, Kaggle **0.597**)

## Hypothesis

exp_012's rescue numbers, split by segment:

| Segment | Δ local | Rescue mechanism |
|---|---:|---|
| MCQ | **+4.80pp** | Trivial — model just emits a single letter |
| Free-form | +0.66pp | Hard — model needs to converge on a numeric answer |

The 96 free-form rescue candidates only yielded ~5 correct answers (5% hit rate). Two competing hypotheses for why:

1. **Token budget too tight.** Stage-2 had `max_tokens=2048` to read the truncated GRPO CoT AND finish thinking AND emit `\boxed{}`. Base Thinking-2507 typically uses 2000–4000 tokens *just for thinking* on hard math problems. With 2048, it likely ran out before converging.
2. **Base model isn't smart enough.** Even with more tokens, Thinking-2507 just can't reason its way to the answer that the larger GRPO trace couldn't reach.

If (1) is the dominant cause, doubling `max_tokens` should give a meaningful free-form lift. If (2), we'd need a different rescue model (e.g. the GRPO model itself).

This experiment tests (1) first because it's a one-knob change.

## Change from baseline (exp_012)

| Knob | exp_012 | exp_013 | Rationale |
|---|---|---|---|
| `rescue.max_tokens` | 2048 | **4096** | Room for thinking + boxed answer on free-form |
| `vllm.max_num_seqs` | 32 | 24 | Halve concurrency to compensate for 2× KV per seq |
| Everything else | — | (unchanged) | Detection, prompts, source data all identical to exp_012 |

MCQ rescue is unaffected by the longer budget — the MCQ rescue prompt forces "output ONLY the letter inside \boxed{}, do not show your work." The model exits early regardless of room.

## Expected impact

- If free-form rescue rate goes from ~5% → 20%: roughly +12 more correct free-form questions ≈ **+1.0pp local**
- If 30%: +20 more questions ≈ **+1.8pp local**
- If unchanged (hypothesis 1 wrong): we learn the bottleneck is model capability, not budget — next step would be exp_014 with GRPO model as the rescuer

Worst case: MCQ rescue stays at its exp_012 level (+4.80pp baked in already) → overall floor is exp_012's 0.597.

## Runtime estimate

- exp_012 took ~45 min on Kaggle T4 x2 at max_tokens=2048
- 2× tokens on ~190 candidates, with max_num_seqs halved → ~80–100 min
- Well under Kaggle 9hr cap

## Implementation

**No code changes.** The generic `rescue_notebook.ipynb` at repo root reads `experiments/<RESCUE_EXPERIMENT>/config.json`. To run:

1. Push this experiment dir to main (already done if you're reading this)
2. On Kaggle: refresh `151b-experiments` dataset
3. Open rescue notebook, change one variable: `RESCUE_EXPERIMENT = "exp_013_rescue_long"`
4. Save Version → Save & Run All

Reuses the same `trevorduong/exp-009-grpo-responses` Kaggle dataset that exp_012 used.

## Results

_Fill in after Kaggle run._

| Metric | exp_012 (baseline) | exp_013 | Δ |
|---|---:|---:|---:|
| Local overall | 57.99% | | |
| Local MCQ | 68.27% | | |
| Local free-form | 52.86% | | |
| Kaggle | 0.597 | | |
| Public rescue rate | 70/190 (36.8%) | | |
| Free-form rescues correct | ~5 of ~95 cand | | |

## Conclusion
- [ ] Keep
- [ ] Discard
- [ ] Needs variant — next: try GRPO model as rescuer (exp_014) if longer budget alone doesn't help free-form
