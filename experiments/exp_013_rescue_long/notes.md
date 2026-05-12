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

## Results (2026-05-12)

| Metric | exp_012 (baseline) | exp_013 | Δ |
|---|---:|---:|---:|
| Local overall | 57.99% (653/1126) | **59.15%** (666/1126) | **+1.16pp** |
| Local MCQ | 68.27% | **71.20%** | **+2.93pp** (+11 q on 375) |
| Local free-form | 52.86% | 53.13% | +0.27pp (+2 q on 751) |
| **Kaggle** | 0.597 | **0.607** | **+1.0pp** |
| Public rescue rate | 70/190 (36.8%) | **106/190 (55.8%)** | +36 rescues |
| Private rescue rate | 64/190 (33.7%) | 88/190 (46.3%) | +24 rescues |
| Net correct lift on public | — | +13 q | (+11 MCQ, +2 FF) |
| Hit rate on additional rescues | — | 13/36 = 36.1% | similar to exp_012 |

**Kaggle translation rate:** 0.86 (1.0pp Kaggle / 1.16pp local) — better than the conservative 0.55 projection.

## Hypothesis test outcome

The pre-experiment hypothesis was: *"longer budget will help free-form because base Thinking-2507 ran out of room before converging on a numeric answer."*

**The hypothesis was wrong.** Free-form only gained +0.27pp (+2 q). The +1.16pp overall came almost entirely from MCQ (+2.93pp). What actually happened:

- MCQ prompt says "output ONLY the letter" but Thinking-2507 still runs internal CoT before the boxed letter. At 2048 tokens this CoT sometimes overflowed → no letter emitted. At 4096 the CoT finishes → letter emitted.
- For free-form, the bottleneck is not budget — it's the **model's reasoning ability**. Even with double the tokens, the rescuer (base Thinking-2507) can't derive a numeric answer that the larger GRPO trace couldn't reach.

**Lesson recorded:** free-form rescue is capability-limited, not budget-limited. Future free-form rescue attempts should use a smarter rescuer (e.g. the trained GRPO model itself as the extractor, or 2-pass self-consistency), not just more tokens.

## Conclusion
- [x] Keep — Kaggle 0.607, new best (+1.0pp)
- [ ] Discard
- [x] Needs variant — for free-form rescue: try GRPO model as rescuer (exp_014?) or self-consistency over the rescue pass. The token-budget lever is exhausted.
