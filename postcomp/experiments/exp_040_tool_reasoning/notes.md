# Experiment: exp_040_tool_reasoning

**Date:** 2026-06-05
**Baseline compared against:** exp_018 stage-1 model (`TrevorDuong/qwen3-4b-thinking-grpo-pass2`), **re-measured on `dev.jsonl`** (the historical Kaggle numbers are a different judge/test set and are not a valid baseline here).

> First post-competition experiment. See `postcomp/README.md` (charter) and
> `postcomp/DEVLOG.md` (2026-06-05) for the motivating diagnosis.

## Hypothesis (revised 2026-06-05 — MEASURED ceiling, see DEVLOG)
Free-form accuracy is gated by **output precision**, not reasoning. The judger demands
**~8 significant figures** (`judger.py:759`, rel tol `1e-8`) against gold stored at full
machine precision; the model reasons correctly but **rounds the final number** (often
because the problem says "round to 2 decimal places") and fails. Having the model
delegate the **final numeric evaluation to a Python tool and emit the unrounded
full-precision result** recovers these.

**Measured ceiling** (`analyze_precision_ceiling.py`, re-judged through the real
`auto_judge` path — NOT a hand estimate):
- **(A) prompt-only "don't round"** (substitute the model's own higher-precision in-trace
  value): **4/46** dev FF errors flip (ids 217, 457, 775, 888).
- **(B) tool adds** (exact compute fixes an accurate-but-imprecise hand value): **+5/46**
  (ids 32, 278, 312, 811, 895). [806 excluded — its gold `2.03972` is stored too coarsely
  to match any exact value = broken gold, unwinnable.]
- **Combined ceiling: 9/46 ≈ 20% of dev FF errors** → **+9 FF questions → 54%→63% FF,
  ≈ +4.5pp overall dev (200q)** at the *optimistic* ceiling (perfect code emission, zero
  regression). Realistic gain is **below** this. MCQ untouched (out of scope).

This is more modest than the initial "+10–18pp FF" guess — the hard measurement
corrected it. The remaining 37 errors are multi-part/format (OUT OF SCOPE — competition
exp_035/037/038/039 proved format post-processing is local-only, **0 on held-out** under
this same judge), truncation (~5), or genuine reasoning failures (~6–8, not tool-fixable).

**Why still worth building:** (1) it's a *robust, generalizing* lever — a tool that
computes exactly can't overfit a curriculum (the failure mode that sank the competition
GRPO passes); (2) the sandbox + tool-loop harness is **reused by Phase 2** (external-data
GRPO reward verification); (3) the precision-vs-strict-judge framing is novel.

## Change from baseline
- **Same model** (`qwen3-4b-thinking-grpo-pass2`), **free-form only** (MCQ routes
  through the unchanged stage-1 path — tools help letter-selection little and risk
  format failures on an already-strong segment).
- Built incrementally, cheapest intervention first (see Probe gate):
  - **(A) Prompt-only "full precision":** instruct the model to emit the final numeric
    answer to ≥10 significant figures and **never round**, explicitly overriding any
    "round to N decimal places" instruction in the problem. No tool. Isolates pure
    display-rounding from hand-arithmetic imprecision.
  - **(B) One-shot PAL (the real lever):** model reasons normally, then emits a single
    final ```` ```python ```` block that *computes and prints* the answer; we execute
    it in a sandbox and use the printed full-precision value as the answer (model's
    `\boxed{}` is the fallback). Single generation — no stop-and-resume, no `<think>`
    reentry risk.
  - **(C) Interleaved TIR:** deferred. Only build if (B) wins AND errors show
    *mid-chain* (not final-step) computation needs. (Advisor: interleaved is the
    riskiest variant on a thinking model — `<think>` reentry corruption, model ignoring
    observations — and the mock can't surface either failure.)
- Code: `postcomp/harness/executor.py` (sandbox: subprocess + **timeout AND memory
  cap** via `resource.setrlimit`, no network, SymPy/numpy/math preamble) +
  `postcomp/harness/pal.py` (one-shot extraction) — both unit-tested locally with a
  **mock model** before any GPU time.

## Probe gate (before any full dev run)
1. **Harness green on mock** (local, no GPU): executor + PAL extraction pass unit tests.
2. **Cheap prompt-only run (A)** on dev FF first — it's nearly free and tells us the
   rounding-vs-imprecision split.
3. **Code-emission probe** (DSMLP, ~30 dev FF Qs, **with one worked demo** — instruction-
   only compliance on this finicky model is a bad bet): does Qwen3-4B-Thinking emit a
   *runnable* final ```python block? Measure (a) % with a runnable block, (b) where it
   lands vs `</think>`, (c) final-computation vs scattered. **Gate:** ≥60% runnable,
   else redesign prompt before scaling. Also check the model card / chat template for a
   **native tool-call convention** before fixing the format.

## Dev results
_Fill in after scoring on dev.jsonl (free-form segment is the one that moved)._

| Metric | Baseline | This | Δ |
|--------|---------:|-----:|---:|
| Overall (200) | | | |
| MCQ (100) | | | |
| Free-form (100) | | | |

## Topic movers
_Top 3 topics that improved / regressed._

## Conclusion
- [ ] Keep → promote TIR as the post-comp free-form default; report on frozen `test.jsonl` once.
- [ ] Discard
- [ ] Needs variant — next experiment idea:
