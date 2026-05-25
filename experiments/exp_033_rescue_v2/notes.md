# Experiment: rescue_v2 (retuned rescue stack)

**Date:** 2026-05-24
**Type:** Rescue experiment — uses `rescue_notebook.ipynb` at repo root.
**Baseline (champion):** exp_018_pass2_rescue — Kaggle **0.628**, local 60.39%.

## Hypothesis

The rescue stack in exp_018 was inherited byte-identical from exp_014 (whose source was exp_009 GRPO, not pass-2). Three findings already in the notes argue the rescue is mis-tuned for pass-2's residual error set:

1. **exp_018's own config** flags that the rescuer should swap strict70 → pass-2 "as exp_019 if exp_018 wins." It won, but exp_019 was never that experiment — the swap was deferred.
2. **exp_014's all-78-truncated finding**: at max_tokens=4096, *every* missing_boxed rescue still truncates before reaching `\boxed{}`. The rescuer is budget-starved.
3. **exp_014's "FF rescue is capability-limited" finding**: free-form rescue produced 0 net gain because the wrong_math cases need a better stage-1 model, not more recovery tokens. Skipping FF rescue is free runtime.

This experiment makes those three changes coordinately and tests whether the retuned rescue lifts the board on the *same* stage-1 (exp_017 pass-2 + original prompts) — i.e. it isolates the rescue stage.

## Change from baseline (exp_018)

Stage-1 source: **unchanged** (= `exp_017_pass2_stage1` responses).
Rescue prompts: **unchanged** (`prompts.py` byte-identical to exp_018).

Three rescue-stage deltas:

| Field | exp_018 | exp_033 | Motivation |
|---|---|---|---|
| `rescue.model_id` | `qwen3-4b-thinking-grpo-strict70` | `qwen3-4b-thinking-grpo-pass2` | The deferred "exp_019" step from exp_018's notes |
| `rescue.max_tokens` | 4096 | **6144** | exp_014: 4096 truncates 78/78 cases |
| `rescue.scope` (new) | (all missing_boxed) | **`mcq_only`** | exp_014: FF rescue is dead weight |
| `vllm.max_model_len` | 8192 | **12288** | Required to fit max_tokens=6144 + max_input=3000 + slack |
| `vllm.max_num_seqs` | 24 | **16** | Absorbs larger per-seq KV cache; MCQ-only cuts case count ~75% so runtime stays similar |

Sampling (T=0.1, top_p=0.95, top_k=20) and `max_input_tokens_from_stage1=3000` are unchanged.

## Implementation note — REQUIRED BEFORE FIRST RUN

`rescue_notebook.ipynb` currently rescues every `missing_boxed` case. The `rescue.scope: "mcq_only"` flag is new and requires a small notebook edit — roughly:

```python
scope = config.get("rescue", {}).get("scope", "all")
if scope == "mcq_only":
    candidates = [c for c in candidates if bool(public_by_id[c["id"]].get("options"))]
```

Insert this filter immediately after the missing_boxed candidate set is built (before the vLLM rescue pass). Backwards-compatible: default scope = "all" preserves current behavior, so existing rescue experiments are unaffected.

Make this edit, commit it, then run.

## Plan

1. Make the `rescue_notebook.ipynb` MCQ-only-filter edit; commit + push.
2. Refresh the `151b-experiments` Kaggle dataset version (bug-111 — never skip this).
3. Confirm the `exp-017-pass2-stage1-responses` dataset is attached to the Kaggle notebook (the rescuer reads stage-1 from it).
4. Set `RESCUE_EXPERIMENT = "exp_033_rescue_v2"` in `rescue_notebook.ipynb`; Save & Run All on Kaggle (T4×2).
5. Download `public_responses.jsonl` + `private_responses.jsonl`; score locally and build the submission.
6. Submit to Kaggle; record the board score.

## Success / abort gate (pre-committed, leaderboard)

Board 1σ ≈ 2.3pp on the ~470-q split. Champion floor = **0.628** (exp_018).

| Board result (vs 0.628) | Verdict | Action |
|---|---|---|
| **≥ +0.005 (~0.633+)** | Retune is real (above noise) | Lock as new champion. If exp_031 also wins later, stack: exp_031 stage-1 + exp_033 rescue config as exp_034. |
| **−0.005 to +0.004** | Tied / noise | Keep exp_018 as champion. The retune is no-loss-no-win on this stage-1; defer further rescue work. |
| **≤ −0.005** | Regression | Diagnose per-bucket locally (MCQ rescue rate? FF retention?). Most-likely culprit is the rescuer swap (strict70 → pass-2 changes the output distribution); back it out first as exp_033b before discarding the budget bump. |

Sequencing context (2026-05-24):
- exp_031 (FF-precision dev probe on pass-2) is in flight. If exp_031 dev passes → full board run, then THIS rescue retune layers on top as exp_034 (= exp_031 stage-1 + exp_033 rescue config).
- exp_033 itself stays anchored on exp_017's stage-1 because that's the apples-to-apples comparison vs the 0.628 champion.

## Results

**Important caveat: the MCQ-only scope filter did NOT execute on Kaggle.** Direct evidence: 66 FF responses were modified by rescue (would be 0 if the filter ran). Root cause: the Kaggle notebook running there is an older copy of `rescue_notebook.ipynb` — the scope-filter edit committed to main on 2026-05-24 never reached the Kaggle environment because the notebook file is uploaded/maintained separately from the `151b-experiments` dataset.

**What actually ran:** pass-2 rescuer + max_tokens=6144 on **all** missing_boxed (both MCQ and FF), not the intended MCQ-only filter. So 2 of the 3 designed changes landed; the scope filter was inert.

Despite this, the result is a clean improvement over exp_018:

| Segment | exp_017 stage-1 | exp_018 (champion) | exp_033 (as-run) | Δ vs exp_018 |
|---|---:|---:|---:|---:|
| MCQ | 63.73% | 73.87% | **77.60%** | **+3.73pp** |
| Free-form | 53.40% | 53.66% | 54.19% | +0.53pp |
| Overall | 56.84% | 60.39% | **61.99%** | **+1.60pp** |

**Reading:** MCQ rescue lifted +3.73pp, exactly the lever we predicted. FF rescue produced only +0.53pp despite better rescuer + more budget — confirming exp_014's "FF rescue is capability-limited" finding even on a stronger rescuer. The MCQ-only scope was an efficiency optimization, not a score lever, so its absence didn't cost score (it would have cut runtime but left score effectively unchanged).

Responses changed by rescue: 143 / 1126 (77 MCQ + 66 FF).

## Conclusion

Best local rescue result yet (+1.60pp over the 0.628 champion). Mechanism is consistent with the documented findings: pass-2 rescuer + longer budget unlocks meaningful MCQ recovery on top of pass-2 stage-1. Submit and board-test against the pre-committed gate.

Projected board lift if rescue transfers at the historical ~50% rate (exp_018: +0.030 local stage-1 → +0.014 board): **~+0.8pp board → ~0.636**. Comfortably above the +0.005 promotion threshold if transfer holds.

**Follow-up needed regardless of board result:** make sure the next rescue experiment uses the up-to-date `rescue_notebook.ipynb` (the version on main with the scope filter) — this run's scope-filter inertness was a stale-notebook issue, not a code bug. Logged as bug-112.

## Board result (2026-05-25 00:06 UTC)

**Board: 0.625** (−0.003 vs the 0.628 champion). Lands in the **tie/noise band** of the pre-committed gate (±0.005 of 0.628 = 0.623–0.633). exp_018 stays as champion.

| Local Δ vs exp_018 | Board Δ vs exp_018 | Transfer |
|---:|---:|---:|
| +1.60pp | −0.30pp | inverted |

Another lossy local→board transfer in the GRPO-flavored direction (see [[project_grpo_local_no_transfer]]). The MCQ +3.73pp local was driven by the pass-2 GRPO model acting as rescuer — and pass-2 GRPO has shown the same overfit-to-public-MCQ pattern at the stage-1 level (pass-3 +1.77pp local → 0pp board; pass-5 +1.68pp local → −1.4pp board). The pattern now appears to be GRPO-model-as-source rather than GRPO-model-as-stage-1 — wherever the pass-N model touches MCQ output, local lifts inflate vs the board.

**Strategic takeaway:** the rescue stack appears saturated on the board. exp_018's 0.628 is likely the *rescue-stage* ceiling for this stage-1 base. Future board gains must come from stage-1 capability levers (extended thinking budget, FF-precision prompts, SFT v2) or from a fundamentally different rescuer mechanism (not GRPO-flavored).
