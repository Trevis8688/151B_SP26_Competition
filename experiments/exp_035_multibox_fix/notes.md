# Experiment: multibox_fix (judger extraction post-process)

**Date:** 2026-05-26
**Type:** Submission-time post-processing fix on top of exp_018. NO model change, NO prompt change.
**Baseline:** exp_018_pass2_rescue (Kaggle 0.628 publicLB).
**Status:** Complete. publicLB tie (0.628), privateScore TBD on 2026-06-01.

## Origin

After exp_034 SFT v2 failed, I declared exp_018 as final. User pushed back: had I actually done error analysis on the 0.628 model's wrong answers? I had not — only run `scripts/analyze.py` for bucket counts. A 30-minute read of exp_018's `public_responses.scored.jsonl` immediately surfaced this:

The judger's `extract_all_boxed` (judger.py:428) only takes the **last contiguous group** of `\boxed{}` expressions. Multi-part responses where the model writes boxes separated by prose (e.g. "Part i): $\boxed{A}$ - Part ii): $\boxed{C}$ - Part iii): $\boxed{0.0050}$") collapse to a single trailing box, and the gold-vs-pred length mismatch scores them wrong even when the model wrote the correct answer in every part.

Example: id=508 (BlueSky Air, 7-part). Gold = `['A','C','0.0051','DG','A','D','D']`. Model's response contains all 7 in order, but extraction returns `D` only.

## Strict rule (verified)

If at submission time:
1. `[ANS]` count in question ≥ 2 (multi-part question), AND
2. Total `\boxed{}` in response ≥ `[ANS]` count, AND
3. Judger's current extraction returns fewer items than `[ANS]` count,

then append a `\n\nFinal Answer: \boxed{...} \boxed{...} ...` block at the very end of the response, containing the LAST N boxes the model wrote (N = `[ANS]` count). This creates a contiguous trailing group the judger extracts cleanly.

`scripts/apply_multibox_fix.py` implements the rule.

## Why the rule is safe (zero local breaks)

- MCQ is skipped (rule does not apply to single-letter answers).
- Condition #3 prevents clobbering responses where the judger is already finding enough answers — currently-correct multi-part responses are never touched.
- `[ANS]` count predicts gold length with 97.6% accuracy on public (733/751 FF questions). The few mismatches are 1-`[ANS]` questions with list-style gold, which the rule leaves alone (n_expected=1 → no-op).

## Results

### Local public (1126 questions)
| Segment | exp_018 | exp_035 | Δ |
|---|---:|---:|---:|
| MCQ (375) | 73.87% | 73.87% | 0 (MCQ skipped) |
| Free-form (751) | 53.66% | **54.73%** | **+1.07pp** |
| Overall (1126) | 60.39% | **61.10%** | **+0.71pp** |
| Modified | — | 50 responses | — |
| Broke (correct→wrong) | — | **0** | — |
| Recovered (wrong→correct) | — | **8** | — |

### Kaggle board (private 943)
| Stage | exp_018 | exp_035 | Δ |
|---|---:|---:|---:|
| publicLB visible | 0.628 | **0.628** | 0 |
| privateLB (held back) | TBD | TBD | TBD until 2026-06-01 |
| Private responses modified | — | 37 | — |

The publicLB tie is consistent with the local-vs-board math: the LB-visible subset is ~470 questions; even at our public modification rate of 4.4%, only ~3-4 LB-visible questions get modified, and only ~16% of modified responses recover. Expected LB-visible flips ≈ 0.5-1, well below the 1σ noise on `0.001` granularity.

## Why we keep exp_035 as the active submission

- Lower-bound is exp_018 (publicLB tie confirms no regression).
- 37 modified private responses give a non-zero chance of a modest privateScore lift; the fix is distribution-free, so private should behave similarly to public.
- Zero risk of the local↔board inversion pattern that killed exp_029/exp_031/exp_033 — this fix is post-processing, not model/prompt-tuning, so the public.jsonl overlap with GRPO training is irrelevant.

## Followups

- exp_036 (verification-pass probe) is scaffolded as a separate, mechanism-different attempt.
- After June 1 deadline: check whether privateScore differs from publicLB by anything visible, to retroactively confirm the fix did anything on the held-back set.
