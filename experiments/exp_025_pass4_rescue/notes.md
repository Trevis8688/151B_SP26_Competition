# Experiment: pass4_rescue

**Date:** 2026-05-22
**Baselines:**
- exp_018_pass2_rescue — pass-2 stage-1 + exp_014 rescue → Kaggle **0.628** (current best), local 60.39%
- exp_024_pass4_stage1 — pass-4 stage-1 only → Kaggle **0.600**, local 60.75%

## Hypothesis

Pass-4 stage-1 reached **0.600** on the board (+1.4pp over the 0.586 pass-2/pass-3 floor — the first GRPO stage-1 board gain). exp_018's rescue stack added **+0.042** on top of pass-2's 0.586 stage-1 (→ 0.628). If pass-4's stronger stage-1 takes a comparable rescue lift, the stack lands ~0.64 — a new best.

This is the cheapest test: a **single-variable swap from exp_018** (`source_experiment` exp_017 → exp_024). Rescue model (strict70), prompts, token budgets, and vLLM sizing are byte-identical to exp_018.

### Realistic expectations / risk

- Pass-4 stage-1's +1.4pp is only ~0.6σ on the ~470-q board split — at the edge of noise. So the +0.042 rescue lift may not fully stack on a stage-1 gain that is itself shaky.
- exp_014/exp_018 lesson: this rescuer lifts **MCQ** (+1.33pp local) but is **saturated on free-form** (0 net). Pass-4's stage-1 gain was broad (MCQ +3.46pp AND FF +1.47pp local), so the parts of pass-4's gain in free-form may not benefit from rescue.
- Pass-4 stage-1 missing_boxed = 9.1% local (103 q) — rescue headroom exists but is modest.

## Change from baseline (exp_018)

**Single variable:** `source_experiment` `exp_017_pass2_stage1` → `exp_024_pass4_stage1` (and the matching `stage1_dataset_name`). Rescuer model, prompts, max_tokens=4096, T=0.1, top_p=0.95, top_k=20, max_input_tokens_from_stage1=3000, vLLM sizing — all identical to exp_018.

## Plan (rescue lifecycle)

1. **Upload exp_024's responses as a Kaggle dataset** named `exp-024-pass4-stage1-responses` (must contain `public_responses.jsonl` + `private_responses.jsonl` from exp_024). This is what `stage1_dataset_name` points at.
2. Commit + push this exp_025 folder; refresh the `151b-experiments` Kaggle dataset.
3. Open `rescue_notebook.ipynb` (repo root or this dir) on Kaggle → set `RESCUE_EXPERIMENT = "exp_025_pass4_rescue"` → attach the stage-1 dataset + utils dataset → Save & Run All (T4×2).
4. Download `public_responses.jsonl` + `private_responses.jsonl` (post-rescue) into this dir.
5. Score local: `~/miniconda3/envs/my-virtenv/bin/python scripts/score.py experiments/exp_025_pass4_rescue/public_responses.jsonl --out experiments/exp_025_pass4_rescue/results.jsonl`.
6. Build submission + submit:
   ```
   ~/miniconda3/envs/my-virtenv/bin/python scripts/build_submission_from_responses.py \
     experiments/exp_025_pass4_rescue/private_responses.jsonl \
     experiments/exp_025_pass4_rescue/submission.csv
   ```

## Success / abort criteria

| Kaggle (vs exp_018 0.628) | Interpretation | Action |
|---|---|---|
| > 0.628 convincingly (≥ ~0.635) | Pass-4 stage-1 gain compounded with rescue → new best | Submit as final; consider swapping rescuer to pass-4 as a follow-up |
| 0.620–0.632 (≈ tie) | Rescue lift didn't stack on the shaky stage-1 gain | Keep exp_018 as best; pivot to rescue-stage diagnosis (why exp_018 > exp_021) |
| < 0.620 | Rescue regressed vs exp_018 | Discard; investigate input mismatch (rescuer fed pass-4 inputs) |

## Results

_(to be filled after rescue run + submission)_

| Metric | exp_018 (pass-2+rescue) | exp_025 (pass-4+rescue) | Δ |
|--------|---:|---:|---:|
| Kaggle | 0.628 | TBD | — |
| Local overall | 60.39% | TBD | — |
| Local MCQ | — | TBD | — |
| Local free-form | — | TBD | — |

## Conclusion

_(to be filled)_
