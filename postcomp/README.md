# Post-Competition Exploration

> **The competition is over.** It concluded 2026-06-01 (private leaderboard revealed).
> Everything in this `postcomp/` directory is *independent continued research* done
> **after** the competition closed. It is deliberately walled off from the frozen
> competition record so the two are never confused.

## What is frozen vs. what is here

| | Location | Status |
|---|---|---|
| **Competition record** | `experiments/exp_000`–`exp_039`, `experiments/log.jsonl`, the Experiment State table in `CLAUDE.md` | **FROZEN.** Do not edit. This is the historical record of the graded competition. |
| **Post-competition work** | everything under `postcomp/` | Active. New experiments continue the `exp_040+` numbering. |

## Why this exists

The graded competition is finished — final standing **0.581 private, rank 59/106**,
on the `exp_018` champion content (GRPO pass-2 stage-1 + a GRPO rescuer stage-2).
It was later found that **many top teams cheated** (answer lookup tables = leakage;
covertly giving the model calculators/tools). So the headline `0.774` winning score
is a tainted target, not a clean ceiling.

The goal here is **not** to chase that number. It is to **build the best principled
system I can** on this task — something *impressive and robust* — and to learn the
levers that genuinely move a small reasoning model's math accuracy.

## The diagnosis that motivates this work

The competition exhausted every *post-process* and *test-time* lever (multibox,
cascade, best-of-N, self-verification) — the private reveal proved each added exactly
**0** on held-out data. The one capability lever never properly pulled was the
**training data**:

> Every GRPO pass in the competition trained on a **~70-prompt curriculum sampled
> from `public.jsonl`** (filtered to the 1≤correct≤3 difficulty band). That is two
> orders of magnitude too small *and* public-derived — which is exactly why deeper
> passes overfit the curriculum and **inverted** on the private set
> (pass-2 0.581 > pass-4 0.572 > pass-3 0.522).

So the "deeper GRPO overfits / local doesn't transfer" lesson is not a law of GRPO —
it is a symptom of a tiny, public-derived curriculum. See `DEVLOG.md` (2026-06-05)
for the full diagnosis.

## The plan

Model constraint is kept: **`Qwen3-4B-Thinking-2507` only** (faithful to the
original competition; a small model also benefits most from tools).

| Phase | Lever | Why |
|---|---|---|
| **1 — Tool-Integrated Reasoning (TIR)** | Let the model write & execute Python+SymPy for the arithmetic/algebra it is bad at. | Attacks the dominant error class (free-form `wrong_math`). Generalizes by construction — a tool that solves the equation can't overfit a curriculum. This is the *legitimate, open* version of what the cheaters did covertly. |
| **2 — External-data GRPO** | Redo RL on a large, *diverse* external verifiable-reward corpus (DeepScaleR / NuminaMath / OpenR1), with a fixed reward recipe. | The one unexhausted *training* lever. External, diverse data → no public overfit → the inversion should not recur. |

## Robustness discipline (mandatory — there is no private leaderboard anymore)

The local score is *exactly what burned the competition* (local rose while private
fell). So "robust" here means **methodology**, not a headline number:

- **`data/splits/test.jsonl` (926q) is FROZEN held-out.** Never train on it, never
  tune on it, never look at its per-item errors during development.
- **Iterate on `data/splits/dev.jsonl` (200q)** only.
- **Report on `test.jsonl` once**, at the end of a phase.
- Scoring uses the existing competition judger (`judger.py` / `scripts/score.py`) so
  numbers are comparable to the frozen record.

Baseline to beat = the `exp_018` stage-1 model, **re-measured on this same dev set**
(not the historical Kaggle number — those are under a different judge and test set).

## Layout

```
postcomp/
  README.md      ← this file (the charter)
  DEVLOG.md      ← dated engineering journal; the development narrative
  experiments/   ← post-comp experiments, continuing exp_040+
  harness/       ← shared post-comp code (sandbox executor, TIR loop, eval)
```

## How to measure (local, no Kaggle)

```bash
# score any responses file against public.jsonl ground truth
~/miniconda3/envs/my-virtenv/bin/python scripts/score.py \
    <responses.jsonl> --out <results.jsonl>
```

Heavy model inference runs on **DSMLP A5000** (the user's Mac cannot host the 4B
model). See `CLAUDE.md` → DSMLP Runtime for the launch pattern.
