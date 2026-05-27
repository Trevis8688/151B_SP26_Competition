# Experiment: verification_probe (design only — not yet launched)

**Date:** 2026-05-26
**Type:** Small-N feasibility probe on local public set. NO Kaggle slot, NO model retraining.
**Baseline:** exp_018 stack (Kaggle 0.628 + multi-box fix exp_035 in flight).
**Status:** scaffolded; small-N probe pending.

## Question this probe answers

Can Qwen3-4B-Thinking-2507 (specifically the `qwen3-4b-thinking-grpo-pass2` checkpoint exp_018 uses for stage-1) **reliably distinguish its own wrong answers from its own right answers** when shown the question + its earlier answer and asked to verify?

If YES (signal > random), we have a new mechanism that's structurally different from prompt-tuning, GRPO retrain, and rescue — it's a *filter*, not a generator. The local↔board inversion that killed exp_029/exp_031/exp_033 came from regenerating answers in ways that overfit public-set residuals; a verification filter only **removes** answers it judges wrong, then we re-extract from the original (or fall back to a rescue). Different mechanism, different overfit risk.

If NO (near random), verification doesn't work on this model. Move on. The probe is the only honest way to find out without burning a full Kaggle slot.

## Probe design (small-N, ~150 questions, local-only)

### Sample selection (from exp_018 public_responses.scored.jsonl)
- 50 random wrong_math FF responses (the target population — currently scored 0)
- 50 random correct FF responses (the control — currently scored 1; must NOT flip these)
- 50 random correct MCQ responses (the secondary control — verification shouldn't break working MCQ either)

Total: 150 responses. ~30-50 min on Kaggle T4×2 with vLLM, or ~1h on DSMLP A5000.

### Verification prompt (v1)

Inputs: `question` + `proposed_answer` (the model's earlier `\boxed{}` content).

```
You previously solved this problem and proposed an answer. Independently verify
whether the proposed answer is correct.

Problem:
{question}

Proposed answer: {proposed_answer}

Carefully redo the key step or recompute the result. Then state:
- VERDICT: CORRECT or INCORRECT
- (If INCORRECT) corrected_answer: {value or "uncertain"}

Wrap your final VERDICT in \boxed{}. If you say INCORRECT and have a confident
corrected answer, also wrap that in \boxed{} after the VERDICT.
```

Crucial design choices:
- The model sees its own work indirectly (the answer) but not its prior chain of thought — so it must re-derive rather than rationalize. Avoids the known self-confirmation trap.
- Asks for a re-computation, not a re-read.
- Requires a structured VERDICT we can extract; avoids open-ended self-reflection.

### Decision rule for the FULL stage (NOT this probe — only relevant if probe succeeds)

If verifier says CORRECT → keep original (no change).
If verifier says INCORRECT AND has a corrected_answer that survives a self-second-look → swap in corrected.
If verifier says INCORRECT but no confident corrected_answer → fall back to exp_014 rescuer's output (or keep original if no rescue candidate).

This 3-way rule is what would go on a full Kaggle stage, IF probe shows signal.

## Probe success criteria (pre-committed)

Confusion matrix on the 150 samples:

|              | True wrong (50) | True correct (100) |
|--------------|----------------:|-------------------:|
| Said INCORRECT | TP (want high)  | FP (want low)      |
| Said CORRECT   | FN (low ok)     | TN (want high)     |

| TPR (TP/50) | FPR (FP/100) | Interpretation | Action |
|---|---|---|---|
| ≥ 40% | ≤ 10% | Real signal — verifier finds 40% of errors while flagging only 10% of correct as wrong | **PROCEED:** design full Kaggle stage with the 3-way decision rule |
| ≥ 25% | ≤ 5%  | Marginal signal — verifier is precise but misses many errors | **PROCEED with caution:** small Kaggle slot, monitor FP risk |
| < 25% OR FPR > 15% | — | Verifier confused / hallucinating / generates more noise than signal | **STOP:** verification doesn't work on this model. Lock exp_018+exp_035. |

The TPR/FPR thresholds are calibrated to the loss math: each flipped-correct-to-wrong costs ~0.001 on local; recovering a wrong is worth ~0.001. We need a clear net positive (TPR > FPR by 4×, given the imbalance: 50 errors vs 100 correct in the probe; in the real exp_018 distribution it's ~327 wrong_math FF vs ~403 correct FF, similar ~1:1.2 ratio).

## What lives in this dir

- `notes.md` — this file (design + decision tree)
- `prompts.py` (TODO) — verification prompt + few-shots if needed
- `probe_notebook.ipynb` (TODO) — Kaggle / DSMLP notebook to run the 150-sample probe
- `probe_responses.jsonl` (output) — verifier output per sample
- `probe_metrics.json` (output) — TPR/FPR/confusion matrix

## When to run

After exp_035 (multi-box fix) board result is known. Order:
1. exp_035 lands → if it lifts, the 0.628 floor moves up too; the verification probe is still worth running because it's a different mechanism.
2. If exp_035 ties or regresses (within 1σ) → still proceed with the probe; we have ~4 days, this is cheap diagnostic work.

## Risks

- **Self-confirmation bias.** Most LLMs are bad at saying "I was wrong." Mitigation: prompt asks for re-derivation, not self-critique. Tests will reveal whether the mitigation is enough.
- **False positives drop currently-correct answers.** Mitigation: probe measures FPR explicitly; STOP threshold is ≤15%.
- **The verifier mode-locks** to one verdict (always says CORRECT or always says INCORRECT). Diagnostic: check verdict distribution before computing TPR/FPR.
