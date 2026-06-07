# Post-Competition Development Log

A dated engineering journal for the post-competition research. Newest entries at
the bottom. Each entry records what was decided/done, *why*, the evidence, and the
next step — enough that a cold reader (or future me) can reconstruct the reasoning.

---

## 2026-06-05 — Kickoff: diagnosis + direction

### Context
The competition closed 2026-06-01. Final standing **0.581 private, rank 59/106** on
the `exp_018` champion (GRPO pass-2 stage-1 + GRPO rescuer stage-2). It was then
learned that **many top teams cheated** — answer lookup tables (pure leakage) and
covertly giving the LLM calculators/tools. So the `0.774` winning score is tainted
and is **not** a target. New goal: build the best *principled* system possible —
something impressive and robust — independent of the leaderboard.

### Diagnosis (the motivating finding)
Reviewed the actual GRPO training setup to locate the real gap:

- `experiments/exp_009_grpo/config.json` → model `qwen3-4b-thinking-grpo-strict70`,
  note: *"path-C strict 70-prompt curriculum."*
- `scripts/filter_curriculum_v2.py` → curriculum = public prompts filtered to
  `1 ≤ num_correct ≤ 3` (the "sweet spot" difficulty band).
- `experiments/exp_015_grpo_pass2/train_grpo.py` → train set built as
  `public.jsonl ∩ SWEET_IDS`, dev IDs excluded. Reward = correctness + a
  `length_bonus` hack added specifically to break `reward_std=0`.

**Conclusion:** every GRPO pass trained on **~70 prompts sampled from `public.jsonl`**.
That is (a) two orders of magnitude smaller than how math RL is actually done, and
(b) public-derived, so it overfits the public distribution by construction. This is
the mechanism behind *both* the competition's central mystery — the local↔private
**inversion** (pass-2 0.581 > pass-4 0.572 > pass-3 0.522 on private, while *local*
kept rising through pass-4) — and the gap to a clean ceiling. The
`length_bonus` band-aid is a second tell: the reward landscape was degenerate
(all completions landing in identical reward buckets), a sign the curriculum gave
too little signal.

So "deeper GRPO overfits / local doesn't transfer" is **not a law of GRPO** — it's a
symptom of a tiny public-derived curriculum.

### Levers reconsidered (post-cheating-reveal)
Every post-process / test-time lever was already proven inert on private during the
competition. The reveal that top teams used calculators reframes one lever as the
clear lead: **tool use, done honestly.** The cheaters validated that tools help; the
only difference is honesty (covert leaderboard inflation vs. open tool-augmented
reasoning). It also directly attacks our dominant error class — free-form
`wrong_math` (arithmetic/algebra slips a 4B reasoning model makes constantly, that a
Python/SymPy interpreter would not).

Lever ranking going forward:

| Lever | Ceiling | Risk | Verdict |
|---|---|---|---|
| **Tool-Integrated Reasoning (TIR/PAL)** | High | Low | 🥇 Phase 1 — robust, generalizes, attacks `wrong_math` |
| **External-data GRPO** | High | High | 🥈 Phase 2 — the one unexhausted *training* lever |
| Self-consistency @ N=16–32 | Low | Low | Cheap door-closer (prior "mode-locked" note says it likely won't help) |
| Post-process / SFT / verification / best-of-N | ~0 | — | Dead — proven inert/regressive in the competition |

### Decisions
1. **Sequence:** Phase 1 (TIR) → Phase 2 (external-data GRPO). (User choice.)
2. **Base model:** keep `Qwen3-4B-Thinking-2507`. (User choice — faithful to the
   original constraint; small model benefits most from tools.)
3. **Separation:** all post-comp work in `postcomp/` on `main`; competition record
   frozen. Continue `exp_040+` numbering. (User choice.)
4. **Robustness discipline:** freeze `test.jsonl` (926q), iterate on `dev.jsonl`
   (200q), report on test once. The local number misled us before; sound
   methodology is the robustness story now. See `README.md`.

### Phase 1 design (TIR)
Interleaved tool use (not one-shot PAL), because the model is a *thinking* model:

```
question → model reasons in NL → emits a ```python block → vLLM STOPS at the
fence → sandbox executes (SymPy/numpy/math preloaded) → append "OBSERVATION:
<result>" → resume → repeat (cap ~4 calls / token budget) → \boxed{} → judge.
```

Two commitments that keep it robust:
- **Inject the model interface** (`solve_tir(question, options, generate_fn,
  execute_fn)`) so the loop + sandbox are fully unit-testable locally with a *mock*
  model — every harness bug is caught before spending A5000 time.
- **Sandbox = timeout-bounded subprocess** with output caps + a math preamble.

The one genuine unknown is empirical: *does Qwen3-4B-Thinking reliably emit
executable code when prompted?* That is the first DSMLP probe (~30 dev free-form Qs),
run only after the harness is green on the mock.

### Next step
Scaffold `postcomp/experiments/exp_040_tool_reasoning/`, build
`postcomp/harness/{executor.py, tir.py}`, prove them on a local mock, then the DSMLP
code-emission probe.

---

## 2026-06-05 (cont.) — Error analysis overturns the TIR mechanism: it's PRECISION, not arithmetic

Before building anything, sized the prize by hand-reading **all 46 free-form errors
exp_018 makes on `dev.jsonl`** (dev FF acc = 54%). Source:
`experiments/exp_018_pass2_rescue/results.newjudge.jsonl`. The hypothesis ("TIR fixes
arithmetic blunders") was **wrong about the mechanism**. The real picture:

**The judger demands ~8 significant figures.** `judger.py:759` →
`abs((pred - round(gold,6))/gold) <= 1e-8`. No rounding of the prediction. Gold
free-form answers are stored at **full machine precision** (13–15 sig figs, e.g.
`7091.66666666667`, `0.498401157310035`). The model reasons correctly, then **rounds
the final number to 2 decimals** — often *because the problem literally says "round to
two decimal places"* (ID 806 computes 2.039716 then writes 2.04 as instructed) — and
fails the 1e-8 check. Verified directly:

```
pred=2.04            gold=2.03972            -> FAIL   (model rounded 2dp)
pred=7091.67         gold=7091.66666666667   -> FAIL
pred=7091.666666667  gold=7091.66666666667   -> PASS   (full precision)
pred=0.498401157     gold=0.498401157310035  -> PASS
```

So this is an **output-precision** problem, not a reasoning problem. TIR is still the
right tool — a Python computation emits the unrounded machine-precision value — but the
*mechanism* is precision, and that's a lever **no competition experiment ever touched**
(prompts, GRPO, rescue, multibox — none address output precision).

**Bucketed the 46 dev FF errors** (auto-classifier + hand reconciliation):

| Bucket | ~count | Tool/precision fixes? |
|---|---:|---|
| Precision-only (right value, over-rounded) | ~12–16 | ✅ compute exact, emit unrounded |
| Multi-part format / mis-boxed (several *already correct*) | ~14 | ❌ — competition proved this is local-only, **inert on private** |
| Truncation (out of token budget) | ~5 | ◐ more budget only |
| Genuine value-wrong / conceptual | ~6–8 | ◐ some (940 regression, 673 exact CDF); not 711/482/21 |

Verified 7 of 8 clean precision cases have high-precision gold a tool can match; only
ID 806's gold (`2.03972`) is stored too coarsely to ever hit (broken gold — ignore).

### Reframed Phase 1 thesis
> Free-form accuracy is gated by **output precision** against an 8-sig-fig judge, not
> by reasoning. ~25–35% of FF errors are a correctly-reasoned but rounded/approximated
> number. Fix: delegate the **final numeric evaluation** to a Python tool and emit the
> **full-precision result, unrounded** (explicitly overriding any "round to N places"
> instruction in the problem).

Two interventions, cheapest first:
1. **Prompt-only "full precision, never round"** (≈free): measures how much is pure
   display-rounding vs. the model's hand-arithmetic being imprecise. Captures the easy
   division-type cases; will NOT fix trig/log/power (hand value isn't 8-fig accurate).
2. **TIR / tool-computed final value** (the real lever): fixes both — exact value AND
   full precision.

**Explicitly out of scope:** the multi-part/mis-box bucket. The competition
(exp_035/037/038/039) already showed format post-processing gains locally but adds
**0 on held-out** under this same strict judge. Chasing it would repeat that mistake.

Artifacts: `postcomp/experiments/exp_040_tool_reasoning/dev_ff_errors_sample.txt`
(the 46 traces). Reframed hypothesis folded into that experiment's `notes.md`.

---

## 2026-06-05 (cont.) — Judger verification: is the precision prize a ghost? (No — it's real)

**The risk (advisor-flagged, correctly):** the precision thesis stands or falls on the
judger being authoritative. The 2026-05-27 board update "fixed FF fraction/decimal
false-negatives" — *exactly* the bucket I'm targeting. If the local `judger.py` were a
stale, over-strict copy, the whole +prize would be an artifact of optimizing the
strictest scorer in the room. A prior memory even records that as of 2026-05-27 the
repo's judger was the OLD version with an open "pull new judger.py" action.

**What I checked:**
1. `git log judger.py` → only "Initial commit", but the file is **modified (uncommitted)**.
   The diff is all LaTeX/symbolic normalization — `fix_sqrt`, `arctan`/function-name
   handling, `^[...]`→`^{...}`, bool ordering. **Numeric tolerance (`1e-8`,
   `round(gold,6)`) untouched.** Someone applied the symbolic half of the 5/27 fix.
2. The official judger is **not** fetchable: Kaggle competition files serve only data
   (`private/public.jsonl`, `sample_submission.csv`); no public dataset hit on search.
3. **Behavioral test of the current judger** — the decisive evidence:
   - Representation-equivalence (the documented fix): `1/2`=`0.5`, `\frac{1}{3}`=`0.333…`,
     `\sqrt{2}`=`1.414…`, `2`=`2.0` → **all PASS.** So this judger **has** the new-judge
     adaptive checks. It *is* the new judge (behaviorally).
   - Rounding/precision (my finding): `2.04` vs `2.03972`, `7091.67` vs `7091.6̄`,
     `0.5` vs `0.4984…` → **all FAIL.** The fix did **not** loosen numeric tolerance.
4. **Magnitude cross-check:** the 5/27 fix moved the board +0.032 (~30 questions). A
   broad numeric-tolerance loosening would flip far more (rounding affects ~16% of FF,
   ~100+ questions). The small +0.032 is consistent with *representation-equivalence*
   only — corroborating that tolerance stayed tight.

**Conclusion:** the current `judger.py` is behaviorally the authoritative new judge, and
the **precision lever is real, not a ghost** — rounded answers genuinely fail a judge
that already handles fraction/decimal/sqrt equivalence. Residual uncertainty: I can't
prove byte-identity with the board judger. *Belt-and-suspenders:* if the user has the
official updated `judger.py`, drop it in and re-run the behavioral test — but the
evidence is strong enough to proceed.

**Sizing (hard numbers, real `auto_judge` path):** all 46 dev FF errors are "winnable"
by emitting exact gold (no unparseable-gold / count-mismatch dead-ends at the judge
level). The precision-specific prize is the subset where the model's reasoning *already*
reached the correct value but rounded it (~12–16 by hand-read). The honest way to turn
that into a measured number is the experiment itself, cheapest-first:
intervention **(A) prompt-only "full precision, never round"** on dev FF — and its
**offline proxy**: substitute the model's own highest-precision in-trace value into the
box and re-judge (computable from existing responses, no GPU). That proxy is the next
analysis step before any GPU spend.

### Status / next
- Judger settled. Precision lever validated. Phase-1 plan (A→B, free-form only) intact.
- Build order: `harness/executor.py` (sandbox, GPU-free) → offline proxy for (A) →
  DSMLP code-emission probe → (A) then (B) on dev.

---

## 2026-06-05 (cont.) — Harness built + GPU-free, and the MEASURED ceiling (recalibration down)

**Harness built and green on mock (no GPU spent):**
- `postcomp/harness/executor.py` — sandboxed Python runner. Subprocess + `timeout` +
  `RLIMIT_AS` memory cap (verified *enforced* even on macOS arm64) + socket-blocked
  preamble + sympy/numpy/mpmath(dps=50) batteries. 8/8 self-tests pass, incl. timeout
  kill, error capture, network block, memory cap stopping `[0]*10**10`.
- `postcomp/harness/pal.py` — one-shot PAL glue: extract last ```python block → run →
  assemble judge-ready `\boxed{}` from full-precision stdout, **fallback to the model's
  own box** so PAL can't score below baseline per-item. 11/11 self-tests pass, including
  a real end-to-end sandbox run AND a judge-integration test proving the full-precision
  tool answer **PASSES** the real judger where the rounded one **FAILED** (7091.67 →
  7091.666…). The full pipeline works modulo the one empirical unknown (model emits code).

**Measured precision ceiling (`analyze_precision_ceiling.py`) — replaces the hand guess:**
Re-judged all 46 dev FF errors through the real `auto_judge` path under two simulated
interventions:
- (A) prompt-only "don't round" (model's own in-trace precision): **4/46** flip.
- (B) tool exact-compute adds: **+5/46** (806 excluded as broken gold — verified its
  coarse `2.03972` gold rejects any exact value).
- **Combined ceiling = 9/46 ≈ 20% of dev FF errors** → **+9 FF Q → 54%→63% FF,
  ≈ +4.5pp overall dev**, *optimistic* (perfect emission, zero regression).

**Recalibration (honest):** this is meaningfully below the initial "+10–18pp FF" hope.
The hard measurement corrected an over-optimistic hand estimate — exactly the robustness
discipline working. The precision lever is **real but modest**. It's still worth building
because (1) it's robust/generalizing (can't overfit), (2) the harness is reused by
Phase 2's GRPO reward verification, (3) the framing is novel. But Phase 2 (external-data
GRPO) remains the higher-ceiling capability swing.

**Decision check for the user:** given the measured +4.5pp ceiling, options are (a)
proceed with the cheap prompt-only (A) run first (nearly free signal), (b) go straight to
PAL (B), or (c) reweight toward Phase 2 sooner. Leaning (a)→(b): the GPU probe + (A) run
is cheap and the harness is built; bank the robust win, then Phase 2.

---

## 2026-06-05 (cont.) — Probe authored end-to-end (GPU-free), ready to fire on DSMLP

**Chat-template check (advisor item #3):** fetched the Qwen3-4B-Thinking-2507
`tokenizer_config.json` — it HAS a native tool-call convention
(`<tool_call>{"name",...}</tool_call>` + `<tool_response>`). That's for *interleaved*
multi-turn tools (deferred intervention C). **One-shot PAL doesn't need it** — the model
just thinks then emits one final ```` ```python ```` fence, which `pal.py` already parses.
So: simple fence for (A)/(B); reserve native tool-call for (C) if we ever build it.
Note: the generation prompt always opens `<think>\n`, so code lands after `</think>`
in the answer span (the probe measures `code_after_think`).

**Authored (all GPU-free, syntax-checked, smoke-tested with a mock):**
- `prompts.py` — 4 conditions: `baseline`, `fullprec_A` (prompt-only "never round"),
  `pal_nodemo`, `pal_demo` (+ a FORMAT-ONLY demo using a neutral sqrt(58) value so the
  model can't regurgitate a transferable answer — resolves advisor "include a demo" vs
  memory exp_005 "regurgitates few-shot numbers" by A/B-testing both arms).
- `config.json` — model = `qwen3-4b-thinking-grpo-pass2` (champion stage-1), probe block
  (40 q, seed 42, 9 recoverable forced in), tool limits, A5000 bf16 vLLM.
- `probe_run.py` — the go/no-go mini-eval: generate × 4 conditions, run PAL outputs
  through the real sandbox, judge everything with the repo judger. Reports per-condition
  accuracy, recovery, regression, code-emit %, runnable %, and a GATE verdict
  (≥60% runnable AND net ≥ baseline). Smoke-tested GPU-free: 40-q sample builds, all 9
  recoverable present, message-building + judge correct.
- `scripts/launch_exp040_probe.sh` — DSMLP A5000 batch wrapper, reuses the
  `.venv-difficulty-v2` (vllm 0.8.5 stack), prints the report to `kubectl logs`.

**Why a real mini-eval, not just "does it emit code":** one DSMLP run answers all three
open questions at once (emission, best variant, net accuracy incl. regression) instead
of a code-emission probe followed by a separate accuracy run.

### Status / next
- ✅ Harness (executor + pal) built, GPU-free, all self-tests pass.
- ✅ Precision ceiling measured: ~20% of dev FF errors, +4.5pp overall dev (ceiling).
- ✅ Probe fully authored, GPU-free-tested, committable. Native-tool vs fence decided.
- ✅ **Committed + pushed to `origin/postcomp`** (commit 24b6b2a). Post-comp work is
  isolated on the `postcomp` branch so graders browsing the public repo see only the
  clean `main` (verified: 0 postcomp entries on origin/main). Bundled the new-judge
  `judger.py` (probe requires fraction/decimal/sqrt equivalence). Pre-existing
  uncommitted changes on main (notebooks, notes, log.jsonl) were deliberately NOT
  swept into the commit.
- ⏭ **Handoff: user launches the probe on DSMLP** (no kubectl/launch.sh on the Mac).
  On dsmlp-login:
    ```
    cd ~/151B_SP26_Competition
    git fetch origin postcomp && git checkout -B postcomp origin/postcomp
    bash scripts/launch_exp040_probe.sh
    kubectl get pods                     # find the pod
    kubectl logs -f <pod>                # PROBE REPORT + GATE verdict print here
    ```
  Then paste the report back; we read recovery/regression/runnable% and decide
  whether PAL clears the gate before any full dev run.

---

## 2026-06-07 — Probe ran, OOM fixed, then a 6h JUDGE HANG ate the run. Two-phase rewrite.

### What happened (two failures, one resolved cleanly)
The first DSMLP launch hit a vLLM A5000 OOM at generation 33/160 (**bug-040**):
`max_num_batched_tokens=20480` allocated a 760 MiB MLP buffer the V1-engine
CUDA-graph pools (2.11 GiB) left no room for. Fix (committed ca0bd4f): batched-tokens
20480→8192, util 0.90→0.85, `max_num_seqs` 32→16, `enforce_eager=true`, and set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **That worked** — the re-run
generated all **160/160** (≈32 min, ~430 tok/s).

Then the pod went to `Error` after sitting ~6h. Post-mortem from `kubectl logs`:
the log ends *exactly* at `Processed prompts: 100%` with **total silence** after —
no report, no traceback. That signature (silent, then SIGKILL at the 6h
`K8S_TIMEOUT_SECONDS` ceiling) is a **hang, not a crash** — and an exception is ruled
out because `judge()` already wraps `auto_judge` in try/except (an infinite loop is
not an exception). **`auto_judge` has no internal timeout, and sympy
`simplify`/`equals` can spin forever** on a pathological symbolic comparison built
from model output. `ls probe_*` confirmed the cost: **only `probe_run.py` present** —
no outputs file. The v1 probe wrote `probe_outputs.jsonl` *after* the judging loop, so
the hang destroyed **all 160 generations (32 min of GPU)**. This is **bug-041**.

### Root cause, stated plainly
Two design faults compounded: (1) the expensive GPU output was only persisted *after* a
fragile CPU step; (2) the judge had no kill-switch. Either alone is survivable; together
they turn one slow sympy comparison into a total loss + a 6h GPU burn.

### Fix — split the probe into two phases; add a reusable timeout-safe judge
- **`probe_run.py` (phase 1, GPU):** generate → **checkpoint every completion to
  `probe_generations.jsonl` the instant `llm.generate` returns** → exit. No judging in
  the GPU pod. The expensive work is now durable before anything fragile runs.
- **`probe_judge.py` (phase 2, CPU):** read the checkpoint → sandbox+assemble PAL →
  judge with a **per-item SIGKILL timeout** → stream `probe_outputs.jsonl`
  incrementally → report. Fresh process (no CUDA → forking is safe), GPU-free, so it's
  re-runnable on the login node if it ever dies. **Logs any id that trips the timeout**
  — that id is the sympy-hang culprit we couldn't see when it took down the whole job.
- **`postcomp/harness/safe_judge.py` (new permanent harness):** `judge_safe(pred, gold,
  timeout=20)` runs the judge in a fresh `python -I` **subprocess** and relies on
  `subprocess.run(timeout=)` issuing **SIGKILL** — which survives even a C-level
  (gmpy2/mpmath) hang that `multiprocessing.terminate()`/SIGTERM would ignore. Chosen
  over `multiprocessing` for that robustness *and* to sidestep fork-from-CUDA. This is
  now the standard judge entry point for the full 200q run and any submission build —
  never call `auto_judge` directly on model text again.
- `config.json`: added `tool.judge_timeout_s=20`. `launch_exp040_probe.sh`: runs
  phase 1 then phase 2; documents the standalone phase-2 recovery command.

### Local validation BEFORE re-spending GPU (advisor gate #1)
- `safe_judge.py` self-test on `my-virtenv`: a `while True: pass` worker is **KILLED in
  2.0s** (proves SIGKILL works); real judge path correct/wrong both right; `0.5 == 1/2`
  passes (re-confirms this is the new judge). ALL PASS.
- End-to-end `probe_judge.py` smoke test with a synthetic 4-condition file: baseline
  rounds→wrong, `fullprec_A` recovers, both PAL arms ran the sandbox (runnable 100%,
  used_tool 100%) and overrode the rounded box→correct; recovery/regression/GATE logic
  and report structure all correct; zero judge timeouts. The whole phase-2 pipeline is
  proven on the Mac.

### Lesson (carried into Phase 2)
A judger that can spin forever on model output *is itself a robustness finding* — it
will hang the 200q dev run and any submission pipeline too. The timeout-safe judge
belongs in the standard path, and **expensive GPU output must be checkpointed before any
CPU post-processing touches it.** Both are now baked in.

### Status / next
- ✅ bug-040 (OOM) fixed and confirmed (160/160 generated).
- ✅ bug-041 (judge-hang data-loss) fixed: two-phase + `safe_judge`, validated locally.
- ⏭ **Re-launch on DSMLP** (delete any `Error` pod first to free GPU quota):
    ```
    cd ~/151B_SP26_Competition
    git fetch origin postcomp && git reset --hard origin/postcomp
    bash scripts/launch_exp040_probe.sh
    kubectl get pods && kubectl logs -f <pod>
    ```
  The generation checkpoints first; the judge can no longer hang 6h or lose the run.
  Paste the PROBE REPORT (+ any judge-timeout ids) and we read the gate.
