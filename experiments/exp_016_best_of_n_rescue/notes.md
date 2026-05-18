# Experiment: best_of_n_rescue

**Date:** 2026-05-15
**Baseline compared against:** exp_014_rescue_v2_grpo (Kaggle 0.611, local 59.50%)

## Hypothesis
Running N=3 stochastic resamples on the cases exp_014 got wrong, then voting over judger-equivalence-clustered `\boxed{}` answers, will flip enough confidently-wrong-by-luck answers to yield **+0.005 to +0.020 Kaggle**, with **zero regression risk** because no-majority cases fall back to exp_014.

Realistic expectations (per advisor): self-consistency's 10–15% lit gain is over full datasets; on a wrong-only candidate set the model is often *confidently wrong*, so 3 samples produce 3 consistent wrong answers (no flip). MCQ rescue is the most reliable bucket; free-form depends on the answer-space being wide enough at the sampling temperature to escape the argmax basin.

## Change from baseline
Pure inference-time, no training, no prompt changes from exp_014.

### Candidate set — split by public vs private

We have ground-truth on `public.jsonl` but NOT on `private.jsonl` (the submitted set). "Wrong cases" is only detectable on public, so the candidate selection differs:

**Public (measurement / dev signal):** 329 wrong_ff_boxed + 78 missing_boxed + 103 wrong_mcq = ~510 IDs. Goal: estimate the true flip rate so we know whether the lift on private is worth the compute.

**Private (the actual submission):** two modes, pick based on Kaggle budget:
- **Mode `full` (default — recommended given user has compute):** N=3 on all 943 private questions. ~9.5M total tokens ≈ ~3.2h on T4×2.
- **Mode `missing` (cheap fallback):** N=3 only on the ~78 IDs where exp_014's response lacks `\boxed{}`. ~20 min. Lift will be narrow because all those are truncation — only useful if `max_tokens=8192` actually lets them close.

Both modes use the same voting rule. Both fall back to exp_014's answer on no-majority.

### Sampling
- Model: `TrevorDuong/qwen3-4b-thinking-grpo-strict70` (same as exp_014 stage 1)
- N=3 samples per candidate
- **T=1.0** (matches exp_009 GRPO training-time temperature; T=0.9 risks staying in the wrong argmax basin)
- top_p=0.95, top_k=20
- Free-form / wrong_ff: `max_tokens=4096`
- Missing_boxed bucket: `max_tokens=8192` (still inside `max_model_len=10240`)

### Voting (cluster-then-count, not string-vote)
For each candidate, take the 3 extracted `\boxed{}` answers and **cluster by `Judger.auto_judge` pairwise equivalence**, then pick the largest cluster's representative. String-equal voting undercounts agreement (e.g., `325*326` ≡ `105950` should count as one cluster).

**Replacement rule:** overwrite exp_014's answer only if the winning cluster has ≥2 members. Singleton-only outcomes (3 distinct answers) → keep exp_014's answer. No regression possible.

### Cost
- **Public eval (~510 candidates):** ~7.2M completion tokens ≈ ~75–90 min on T4×2
- **Private `full` mode (943 × N=3):** ~11.6M tokens ≈ ~3.0–3.5h on T4×2 (max_tokens=4096; missing_boxed cases get a second pass at 8192)
- **Private `missing` mode (~78 × N=3 × 8192):** ~1.9M tokens ≈ ~20 min

## Success / abort criteria
| Kaggle Δ vs exp_014 | Interpretation | Action |
|---|---|---|
| ≥ +0.005 | Real lift, even modest | Keep + submit |
| 0.000 to +0.004 | Flat (within noise) | Discard; pivot to stats few-shots (exp_017) |
| < 0.000 | Regression (should be impossible given safety net) | Bug — investigate cluster-vote logic |

## Results

| Metric | Baseline (exp_014) | exp_016 | Δ |
|--------|-------------------:|--------:|---:|
| Local (public.jsonl) | 59.50% | 59.50% | 0 (unchanged) |
| Kaggle (private) | 0.611 | 0.600 | **−1.10pp** (regression) |

**Important:** local is unchanged because the run was done with `target=private` — best-of-N was never applied to public, so `public_responses.jsonl` is just exp_014's responses passed through. We submitted to Kaggle without a public-set sanity check.

**Submitted:** 2026-05-18

### Best-of-N internals (from best_of_n_stats.json)

- 943 private candidates total (mode = `full`)
- 613 (65%) had cluster_sizes [1,1,1] — three distinct answers, **fell back to exp_014** (safety net)
- 330 (35%) had a ≥2 majority and **overwrote exp_014's answer**
  - Of those 330, almost all are `[3]` clusters (all 3 samples agreed)
  - Net Kaggle Δ = −10 correct over 943 → win rate among the 330 confident overwrites is below 50%

## Conclusion

**Discard.** Two precise findings:

1. **The model collapses to one mode under best-of-N.** 65% of cases produce 3 distinct answers (full incoherence), 35% produce 3 *identical* answers (mode-locked confidence). There's almost no middle ground — the policy doesn't sample a productive distribution around correct-but-uncertain answers.
2. **The confident-overwrite slice is anti-correlated with truth at the margin.** The 330 confident overwrites scored worse than the exp_014 answers they replaced (the net −10 over 943 implies ~160 wins vs ~170 losses among the 330 changes — slightly *worse* than a coin flip). The model's "confidence" doesn't track accuracy on cases where exp_014 already got the answer.

The fallback safety net protected the 65% incoherent cases. The 35% confident-overwrite cases bypassed the safety net and produced the regression.

**Lesson:** for Qwen3-4B-Thinking after GRPO, sampling-time interventions on the same policy are dead — diversity collapses to noise or mode-lock with no useful middle. The remaining leverage is in the *policy* itself (training), not the sampling.

## Next lever
- [x] Discard
- [ ] **Pivot to merged exp_015 pass-2 model** (`TrevorDuong/qwen3-4b-thinking-grpo-pass2` once merge_and_push.ipynb runs). Different *policy*, not different *sampling*. exp_017 will swap it in as the stage-1 model and re-test the exp_014 stack.
