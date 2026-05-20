# Experiment: hybrid_route

**Date:** 2026-05-20
**Baselines:** exp_018 (Kaggle **0.628**, current best) · exp_021 (Kaggle **0.611**, public 62.17%)

## Why this exists

exp_021 was the expected new best (public 62.17%, +1.78pp over exp_018) but **regressed to 0.611 on Kaggle** — a private translation of **−0.96×** vs exp_018's +1.91×. The flip was diagnosed locally as a free-form **calibration failure** in pass-3, not a broad regression:

| Slice (public, exp_018 → exp_021) | n | Δ |
|---|---:|---:|
| Overlap-passthrough MCQ | 255 | +2 (85.5% → 86.3%) |
| Overlap-passthrough FF | 642 | +8 (62.1% → 63.4%) |
| Shared-rescue MCQ | 71 | +6 |
| Shared-rescue FF | 65 | +1 |
| Reverse changed-mind MCQ (pass-2 boxed, pass-3 needed rescue) | 17 | **−4** |
| Reverse changed-mind FF | 17 | +1 |
| Changed-mind MCQ (pass-2 truncated, pass-3 boxed) | 32 | pass-3 26/32 (81%) — strong win |
| **Changed-mind FF** (pass-2 truncated, pass-3 boxed) | 27 | **pass-3 3/27 (11%)** — confident wrong |

Public sum: +25 questions. Private inverse: −16. The only meaningfully bad slice is **changed-mind FF**: pass-3 stage-1 produces a free-form boxed answer at 11% accuracy in cases where pass-2 would have truncated and let the rescuer recover (~30% accuracy on the same set). On public this is dominated by the +22 MCQ gain from changed-mind MCQ; on private, MCQ wins generalized less (10% random floor) while the FF over-confidence persisted.

## Hypothesis

Route by question type: MCQ from pass-3, FF from pass-2. The strict70 rescue layer stays — both source stacks (exp_018, exp_021) already include it. No new inference needed; this is a row-level routing of existing responses.

Predicted Kaggle: ≥ 0.628 (worst case = exp_018 if pass-3 MCQ doesn't generalize positively to private; best case = 0.628 + (pass-3 private MCQ gain)). The downside is bounded: hybrid cannot do worse than exp_018 unless pass-3 MCQ *regressed* on private, which would also have shown as a sharp drop on the overlap-passthrough MCQ slice (it didn't — +2 there).

## Change from baseline (exp_018 / exp_021)

For each row:
- `has options` → take exp_021's post-rescue response (pass-3 stage-1 + strict70 rescue)
- `no options`  → take exp_018's post-rescue response (pass-2 stage-1 + strict70 rescue)

No new model, no new inference. Pure submission-build re-routing.

## Plan

1. Build hybrid responses + submission.csv from existing exp_018/exp_021 outputs ✓ (done)
2. Score on public ✓ (61.19%, exactly the sum of exp_021 MCQ + exp_018 FF — routing verified)
3. Submit to Kaggle
4. Re-evaluate pass-4 strategy based on the result (see "Next lever")

## Results

| Metric | exp_018 | exp_021 | **exp_023 hybrid** |
|--------|--------:|--------:|-------------------:|
| Local overall | 60.39% | 62.17% | **61.19%** |
| Local MCQ | 73.87% | **76.27%** | **76.27%** (from exp_021) |
| Local free-form | 53.66% | 55.13% | **53.66%** (from exp_018) |
| Kaggle (private) | **0.628** | 0.611 | _(to submit)_ |

Local hybrid sits *between* exp_018 and exp_021 by construction. The whole bet is that the private translation behaves opposite to public: pass-3 MCQ adds value, pass-3 FF subtracts.

## Conclusion

_(to be filled after Kaggle submission)_

## Next lever

- [ ] **Submit exp_023 hybrid** to Kaggle (1 slot).
- [ ] **Hold exp_022 (pass-4 training) until exp_023 lands.** Pass-4 from pass-3 base would inherit the FF over-confidence problem — if exp_023 confirms it, we should consider re-basing pass-4 on pass-2, or designing the next GRPO pass with a calibration term in the reward.
- [ ] If exp_023 ≥ 0.635: confirms the FF-calibration story; pass-4 strategy = add calibration shaping OR rebase on pass-2.
- [ ] If exp_023 ≈ 0.628: hybrid only matches exp_018, meaning pass-3 MCQ didn't help private either. The pass-3 lift was entirely public-specific; deeper rethink needed before pass-4.
- [ ] If exp_023 < 0.625: something more is going on (rare; would invalidate the slice diagnostic).
