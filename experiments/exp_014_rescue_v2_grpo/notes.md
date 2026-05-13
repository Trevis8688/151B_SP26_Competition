# Experiment: rescue_v2_grpo

**Date:** 2026-05-13
**Baseline:** exp_013_rescue_long (local 59.15%, Kaggle **0.607**)

## Hypothesis

exp_013 ran a clean A/B on rescue token budget (2048 → 4096) and explicitly **disproved** the budget hypothesis for free-form:

| Segment | exp_012 (2048 tok) | exp_013 (4096 tok) | Δ |
|---|---:|---:|---:|
| MCQ rescue | +4.80pp | +2.93pp on top | +7.73pp cumulative |
| Free-form rescue | +0.66pp | +0.27pp on top | **+0.93pp cumulative** |

exp_013's recorded lesson: *"free-form rescue is capability-limited, not budget-limited. Future free-form rescue attempts should use a smarter rescuer."*

This experiment tests that lesson by swapping the rescuer model from base `Qwen3-4B-Thinking-2507` to the exp_009 GRPO model. The GRPO model already showed +3.2pp Kaggle lift over base when used end-to-end (exp_009: 0.583 vs exp_006: 0.551), which means it is meaningfully smarter at deriving the math answer. Whether that capability gain transfers to the rescue-extraction task is the open question.

## Change from baseline (exp_013)

| Knob | exp_013 | exp_014 | Rationale |
|---|---|---|---|
| `rescue.model_id` | `Qwen/Qwen3-4B-Thinking-2507` | **`TrevorDuong/qwen3-4b-thinking-grpo-strict70`** | Test capability hypothesis |
| `rescue.max_tokens` | 4096 | 4096 | Hold constant |
| `rescue.temperature` | 0.1 | 0.1 | Hold constant |
| Rescue prompts | (unchanged from exp_013) | (unchanged) | Hold constant |
| Source data | exp_009 missing_boxed | (unchanged) | Hold constant |
| Detection logic | `"\\boxed" not in response` | (unchanged) | Hold constant |
| vLLM sizing | T4 x2, max_model_len=8192, max_num_seqs=24 | (unchanged) | Hold constant |

**Only `rescue.model_id` changes.** Everything else is identical to exp_013.

## Expected impact (asymmetric)

The rescue is upside-only by design (failed extractions leave the original response, scoring unchanged), so the floor is exp_013's 0.607. The ceiling depends on how the GRPO model behaves on each segment:

### Free-form (the bet)
- exp_013 free-form rescue: 96 candidates, ~5 correct → 5% hit rate
- If GRPO model achieves 15% hit rate on the same 96 candidates: +10 correct → **+0.89pp local**
- If 25%: +20 correct → **+1.78pp local**
- If unchanged (~5%): hypothesis wrong, free-form rescue is fully saturated

### MCQ (the risk)
- exp_013 MCQ rescue is +7.73pp cumulative — most of exp_012/013's gains came from here
- The base model's MCQ rescue prompt is "output ONLY the letter inside `\boxed{}`. Do not show your work." The base model obeys this well enough.
- The GRPO model was trained to do long CoT before answering. At max_tokens=4096 with temperature=0.1, it will likely run a thinking block and *then* emit the letter — same final answer, just slower
- **Risk:** if the GRPO model's thinking block runs past 4096 on some MCQ candidates, it won't emit the letter at all — net MCQ regression
- Mitigation: temperature=0.1 is already aggressive; if the GRPO model also hits the box token after thinking, both models converge

### Net projection
- Best case: free-form +1.5pp, MCQ unchanged → ~0.620 Kaggle
- Likely case: free-form +0.5–1.0pp, MCQ −0.3 to 0.0pp → ~0.610 Kaggle
- Worst case: free-form unchanged, MCQ regresses 1–2pp → ~0.600 Kaggle (still ≥ exp_012's 0.597)

## Implementation

**No notebook changes.** The generic `rescue_notebook.ipynb` at repo root reads this experiment's `config.json` via `RESCUE_EXPERIMENT`. To run:

1. **Pre-flight: confirmed GRPO model is merged on HF Hub.** `TrevorDuong/qwen3-4b-thinking-grpo-strict70` is a single 8GB safetensors (not adapter-only), vLLM-loadable directly.

2. Push this experiment dir to main (committed).
3. On Kaggle: refresh `151b-experiments` dataset.
4. Open `rescue_notebook.ipynb`, set `RESCUE_EXPERIMENT = "exp_014_rescue_v2_grpo"`.
5. Save Version → Save & Run All. Reuses `trevorduong/exp-009-grpo-responses` dataset.

## Runtime estimate

- exp_013 took ~80 min on Kaggle T4 x2 (4096 tokens, max_num_seqs=24)
- Same config here → ~80 min, well under 9hr cap
- GRPO model is the same size (4B), so VRAM usage is unchanged

## Risks

1. **MCQ regression** (largest concrete risk). If GRPO model overthinks MCQ rescue prompts, MCQ accuracy could drop. If observed, next experiment is per-segment routing (base for MCQ, GRPO for free-form).
2. **GRPO model not merged on Hub.** Currently the GRPO repo may be adapter-only — vLLM won't load it directly. See pre-flight step above.
3. **Capability hypothesis wrong.** If free-form rescue stays at ~5% hit rate, rescue-as-a-mechanism is saturated and we need a different lever (best-of-N, hybrid routing, more training data, etc).
4. **Distribution shift.** GRPO was trained on the 70 strict-curriculum prompts from public.jsonl. The missing_boxed candidates are by definition the hard tail. GRPO's specialization may or may not generalize to that tail.

## What success looks like

- Kaggle ≥ 0.612 (≥ +0.5pp over exp_013): hypothesis confirmed → keep, then sweep
- Kaggle 0.605–0.611: ambiguous, treat as flat → next move is per-segment routing
- Kaggle < 0.605: regression → revert to exp_013 as best, pivot to non-rescue lever (best-of-N or fresh fine-tune)

## Results

_(to be filled in after Kaggle run)_

| Metric | exp_013 (baseline) | exp_014 | Δ |
|---|---:|---:|---:|
| Local overall | 59.15% | | |
| Local MCQ | 71.20% | | |
| Local free-form | 53.13% | | |
| Kaggle | 0.607 | | |
| Public rescue rate | 106/190 (55.8%) | | |
| Free-form rescue hit rate | ~5% | | |

## Conclusion

- [ ] Keep
- [ ] Discard
- [ ] Needs variant
