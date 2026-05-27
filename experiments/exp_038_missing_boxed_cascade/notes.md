# Experiment: missing_boxed_cascade (targeted re-inference)

**Date:** 2026-05-27
**Type:** Re-inference on the 95 still-missing-`\boxed{}` private responses after exp_037. Additive-only — replaces ONLY responses without an existing extractable answer.
**Baseline:** exp_037_multibox_v2 (Kaggle publicLB 0.628).
**Status:** Scaffolded; Kaggle run pending.

## Why this lever

After exp_037, **95 of 943 private responses still have no `\boxed{}`** (54 FF + 41 MCQ). All are clear truncation cases — responses end mid-sentence with median length ~24,000 chars (~8192-token cap). These score 0 by definition; any answer extracted is pure upside, zero break risk.

The mechanism is *different* from everything that died:
- Not retraining (GRPO scaling failed)
- Not rescue retune (saturated)
- Not prompt-on-GRPO (board inversion)
- Not SFT v2 (collapse)
- Not self-verification (4% TPR)

This is **targeted re-inference with a different generation strategy** on questions that currently extract nothing. The only requirement to win: produce a `\boxed{}` containing a plausible answer. We don't need to beat the prior model on these — there is no prior answer to beat.

## Why more max_tokens won't help on its own

The 95 cases failed at 8192 max_tokens because thinking exploded. Pushing to 16,384 just buys more thinking, same failure mode at higher cost. The fix is changing the GENERATION STRATEGY, not the budget.

## Cascade design

Each ID receives up to three attempts; the first attempt that produces a `\boxed{}` wins.

| Stage | Target | Strategy | max_tokens | Samples |
|---|---|---|---:|---:|
| 1 | All 95 | Greedy + suppressed-thinking prompt (`<think>\n</think>` inserted) | 2048 | 1 |
| 2 | Stage-1 residual | Standard prompt, T=0.6, first sample with `\boxed{}` wins | 4096 | 4 |
| 3 | MCQ-only residual | Letter-only prompt (`Respond with a single letter A-J in \boxed{...}`) | 512 | 4 |

The suppressed-thinking prompt forces `<think>\n</think>` before the model generates, so it skips the thinking explosion entirely and goes straight to writing the answer. This is a Qwen3-Thinking-specific hack — untested here but the cleanest way to cut the failure at its source.

Stage 2 gives sampling diversity a chance: some chains terminate, others explode, take any chain that finished.

Stage 3 forces an MCQ guess. Even a uniform-random guess on 10 options has 10% recovery rate; a partially-informed forced guess should do meaningfully better.

## Model

`Qwen/Qwen3-4B-Thinking-2507` (base — NOT the GRPO variant). exp_014's lesson: GRPO chains are 2–4× longer than base chains, and these 95 cases failed because of length. Reverting to base buys ~3× tighter chains for free.

## Expected outcome

- MCQ recovery: 10–25% on a wild guess, higher on a forced direct answer = 4–12 questions
- FF recovery: 5–15% if truncation chopped near a final answer = 3–8 questions
- Total estimate: ~7–20 net wins on private = +0.007 to +0.020 board

## Risk profile

Zero break risk. Currently-extracted answers in exp_037 are untouched. Only responses with no `\boxed{}` get replaced. The replacement either (a) extracts a valid answer (upside) or (b) also has no `\boxed{}` (no change).

## What lives in this dir

- `notes.md` — this file
- `targets.jsonl` — the 95 private questions to re-infer (built by build_targets script in this notebook's dir)
- `cascade_notebook.ipynb` — Kaggle T4×2 cascade runner
- `cascade_responses.jsonl` — output (gitignored)
- `submission.csv` — output (gitignored)
