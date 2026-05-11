# Experiment: boxed_rescue

**Date:** 2026-05-11
**Baseline:** exp_009_grpo (local 55.95%, Kaggle **0.583**)

## Why not hybrid routing (the original plan)?

Originally exp_012 was going to route MCQ → exp_004 base, free-form → exp_009 GRPO. Local data refutes the premise: exp_009 beats exp_004 on **both** segments.

| Segment | exp_004 correct | exp_009 correct | Net |
|---|---:|---:|---:|
| MCQ (375 q) | 237 | 238 | +1 for exp_009 |
| Free-form (751 q) | 386 | 392 | +6 for exp_009 |

There is no segment where routing to exp_004 helps. Hybrid routing is dead.

## Hypothesis (boxed_rescue)

**The `missing_boxed` bucket accounts for 16.3% of failures** (183 out of 1126 questions). These responses run out of tokens before the model emits `\boxed{}` — but the reasoning has often *already* converged on an answer; it's just stuck mid-CoT.

A follow-up "extract the boxed answer" prompt — take the truncated response and re-prompt with `"Given the above reasoning, state your final answer in \boxed{}"` at a small token budget (256) — should rescue a meaningful fraction of these. Each rescue is worth ~+0.09pp (1/1126).

**Expected gain:**
- If 50% of missing_boxed cases have a recoverable answer in the truncated CoT: ~+8pp (91 questions × 0.089pp)
- If 25%: ~+4pp
- If 10%: ~+1.6pp

This is the highest-leverage cheap win available: it directly attacks the single largest failure bucket.

## Change from baseline (exp_009)

Two-stage inference:
1. **Stage 1 (unchanged from exp_009):** Generate response with `max_tokens=8192, T=0.6`, save raw.
2. **Stage 2 (new):** For each response that lacks a properly-formed `\boxed{}` after `</think>`, build a new prompt with the original question + the truncated response + an extraction instruction. Generate with `max_tokens=512, T=0.1` (low temp = deterministic extraction).
3. **Merge:** If Stage 2 produced a `\boxed{}`, replace the original response with the truncated Stage 1 + Stage 2 extraction text. Otherwise keep the original.

| Knob | exp_009 | exp_012 | Rationale |
|---|---|---|---|
| Stage 1 generation | (same as exp_009) | (same) | Don't disturb what's working |
| Stage 2 rescue prompt | n/a | NEW | See below |
| Stage 2 temperature | n/a | 0.1 | Extraction should be deterministic |
| Stage 2 max_tokens | n/a | 512 | Short; just needs the box |

### Stage 2 prompt template

```
<original question>

Your reasoning so far:
<truncated stage 1 response, with <think>...</think> stripped>

Based on the reasoning above, what is your final answer? Reply with only \boxed{...} containing the answer.
```

For MCQ: the instruction says `\boxed{LETTER}`.
For free-form: the instruction says `\boxed{value}` (matching the system prompt).

## Implementation

The current Kaggle notebook doesn't support two-stage inference natively. Options:

**Option A (cleaner, more work):** modify `cse151b-notebook.ipynb` to run a stage-2 pass over missing-boxed responses.

**Option B (faster scaffold, recommended):** post-process locally.
1. Run exp_011 (best-of-N) on Kaggle as-is — generates `public_responses.jsonl` and `private_responses.jsonl`
2. Download both files
3. Run a local stage-2 script (`scripts/boxed_rescue.py`) that: (a) finds missing-boxed responses, (b) ships them through HF Inference API or local Qwen3-4B for stage 2, (c) merges results back
4. Rebuild `submission.csv` from the rescued `private_responses.jsonl`

**Caveat for option B:** stage 2 inference happens off-Kaggle, which means we need a way to run Qwen3-4B locally or via API. RTX 3060 (12GB) can fit Qwen3-4B in 4-bit quant for inference — fine for ~183 short generations.

## Runtime estimate

- Stage 1 (Kaggle): same as exp_011 (~6-8 hr if N=3)
- Stage 2 (local RTX 3060 in 4-bit): ~183 questions × ~10 sec/q = ~30 min

So total: <1 day end-to-end if exp_011 already ran.

**Dependency:** exp_012 should run *on top of* exp_011 (best-of-N already reduces missing_boxed below 16.3% by giving 3 shots at producing a box). Run exp_011 first, then apply rescue to whatever's left.

## Dev results

_Fill in after stage 2 runs on exp_011's missing-boxed cases._

| Metric | Baseline (exp_011) | This (exp_012) | Δ |
|---|---:|---:|---:|
| Overall | | | |
| MCQ | | | |
| Free-form | | | |
| missing_boxed count | | | |
| rescues that produced correct answer | | | |

## Risks

1. **Truncated CoT may not contain the answer.** If the model ran out of tokens before even converging on an answer (still exploring branches at token 8192), stage 2 has nothing to extract. Some fraction of the 183 will be unrescuable.
2. **Hallucinated extraction.** Stage 2 might invent an answer that doesn't follow from the truncated reasoning. T=0.1 minimizes this but won't eliminate it.
3. **Token-budget interactions.** If the truncated CoT is very long (close to 8192), the stage 2 prompt itself becomes long — need to truncate stage 1 to last ~3000 tokens (keep tail, where the conclusion lives).

## Files

```
experiments/exp_012_boxed_rescue/
├── notes.md       This file
├── config.json    Stage 2 sampling config
├── prompts.py     Stage 2 prompt template (extraction-only)
└── (planned) scripts/boxed_rescue.py   Stage 2 post-process script
```

## Conclusion
- [ ] Keep
- [ ] Discard
- [ ] Needs variant — next experiment idea:
