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

**Runs entirely on Kaggle via `rescue_notebook.ipynb`** (at repo root, alongside `cse151b-notebook.ipynb`). The notebook is generalized — change one variable (`RESCUE_EXPERIMENT`) and the experiment's `config.json` declares which upstream source to rescue. Reusable for any future experiment with `missing_boxed` failures.

### How to run (one-time setup)

1. **Upload stage-1 outputs as a Kaggle dataset.** From local repo:
   ```bash
   mkdir -p /tmp/exp009-responses && cd /tmp/exp009-responses
   cp ~/CSE151B/151B_SP26_Competition/experiments/exp_009_grpo/public_responses.jsonl .
   cp ~/CSE151B/151B_SP26_Competition/experiments/exp_009_grpo/private_responses.jsonl .
   cat > dataset-metadata.json <<EOF
   {"title": "exp_009_grpo responses", "id": "trevorduong/exp-009-grpo-responses", "licenses": [{"name": "CC0-1.0"}]}
   EOF
   KAGGLE_API_TOKEN="..." ~/miniconda3/bin/kaggle datasets create -p .
   ```

2. **Create a new Kaggle notebook** importing `rescue_notebook.ipynb` from the GitHub repo (or upload manually). Attach inputs:
   - `cse-151-b-spring-2026-competition` (competition data)
   - `trevorduong/151b-experiments` (auto-pulls from main branch, contains exp_012 config)
   - `trevorduong/exp-009-grpo-responses` (the dataset created in step 1)
   - GPU: **T4 x2**

3. **Save Version → Save & Run All** — ~45 min total. Downloads `submission.csv` from output panel.

### Detection logic

Mirrors `analyze.py` exactly: a response "needs rescue" iff `"\\boxed" not in response.response`. This matches the 183 `missing_boxed` count from the scorer's bucket. We do NOT try to rescue responses that already have *some* boxed (even if wrong) — that would risk overwriting a correct extraction with a worse one.

### Merge strategy

Successful rescues are **appended** (not replaced) onto the original response, prefixed with `[RESCUE EXTRACTION]:`. The judger picks up the new box; the original CoT stays for human review. Failed rescues (where stage-2 also didn't emit `\boxed`) leave the original untouched — net effect on score 0.

## Runtime estimate

- Model load: ~10 min (same as stage-1)
- Rescue inference: ~200 candidates × short generations (max 2048 tokens, T=0.1) ≈ **20-30 min**
- Total: **~45 min**, well under Kaggle 9hr session cap

Independent of exp_011 — runs directly on exp_009 outputs.

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
├── config.json    source_experiment + stage1_dataset_name + rescue/vllm params
└── prompts.py     RESCUE_SYSTEM_PROMPT_MATH/MCQ + build_rescue_user_message()
```

Generic notebook lives at repo root:
```
rescue_notebook.ipynb   Reads exp_NNN/config.json for source_experiment + stage1_dataset_name
```

## Stage-2 model decision

**Base `Qwen/Qwen3-4B-Thinking-2507`**, not the GRPO model. Reason: the rescue tool is meant to be reusable for any future experiment. The base is the "common denominator" that works regardless of which model produced the upstream output. The extraction task (read partial reasoning → emit one boxed answer) is simple enough that GRPO's reasoning gains don't apply here.

If a future experiment shows the GRPO model is meaningfully better at rescue, override via `rescue.model_id` in config.json — no notebook changes needed.

## Conclusion
- [ ] Keep
- [ ] Discard
- [ ] Needs variant — next experiment idea:
