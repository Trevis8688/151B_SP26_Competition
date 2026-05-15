# CSE 151B Competition — Project State & Next Steps
**Date:** 2026-04-01  
**Competition Deadline:** ~2 months (early June 2026)

---

## Current State

### What Exists
- **Starter code** (`starter_code_cse151b_comp.ipynb`): Complete end-to-end pipeline using Qwen3-4B-Thinking (INT8 quantized) with vLLM batched inference
- **Scoring infrastructure** (`judger.py`, `utils.py`): Robust auto-grading with 10 judgment types covering symbolic/numeric equivalence via SymPy
- **Datasets**: `public.jsonl` (1,116 questions with answers) and `private.jsonl` (893 questions, no answers)
- **CLAUDE.md**: Full project documentation

### What Has NOT Been Done
- No model inference runs have been executed
- No submission CSVs generated
- No experimental results or baselines recorded
- No prompt variations tested
- No leaderboard submissions made
- The `results/` and `findings/` directories are empty

### Dataset Breakdown
| Set | MCQ | Free-form | Total |
|---|---|---|---|
| Public (with answers) | 375 | 741 | 1,116 |
| Private (no answers) | 300 | 593 | 893 |
| **Total** | **675** | **1,334** | **2,009** |

### Baseline Configuration
- **Model:** Qwen3-4B-Thinking-2507 (INT8 via BitsAndBytes)
- **Sampling:** temp=0.6, top_p=0.95, top_k=20, max_tokens=32768
- **Prompts:** Two system prompts (MCQ selector + free-form step-by-step solver)
- **Output format:** Answer in `\boxed{}`

---

## Recommended Next Steps (Priority Order)

### Phase 1: Establish Baseline (Do First)
1. **Run the starter code as-is** on the public set to get a baseline accuracy score
   - Record MCQ accuracy and free-form accuracy separately
   - Save results to `results/starter_results.jsonl`
2. **Generate a submission** for the private set and submit to Kaggle
   - This gives you a leaderboard baseline to improve against
3. **Analyze errors** — categorize where the baseline fails:
   - MCQ: wrong letter selected vs. formatting issues
   - Free-form: wrong math vs. extraction/formatting errors
   - Which math topics are weakest (algebra, calculus, combinatorics, etc.)

### Phase 2: Quick Wins (Low Effort, Moderate Gain)
4. **Prompt engineering improvements:**
   - Add few-shot examples (2-3 solved problems per type) to the system prompt
   - For MCQ: instruct the model to briefly evaluate each option before selecting
   - For free-form: add explicit instructions for common pitfalls (e.g., "simplify your answer", "convert decimals to fractions where exact")
   - Test chain-of-thought prompting variations
5. **Sampling parameter tuning:**
   - Try temperature=0 (greedy) for deterministic answers
   - Try lower temperature (0.1-0.3) for more focused reasoning
6. **Majority voting (self-consistency):**
   - Generate N=5-10 responses per question
   - Take the most common answer
   - This alone can boost accuracy 5-15% on reasoning tasks

### Phase 3: Model Upgrades (Medium Effort, High Gain)
7. **Upgrade model size:**
   - Qwen3-8B or Qwen3-14B (if GPU memory allows)
   - Qwen3-32B with INT4 quantization
   - Other strong reasoning models: DeepSeek-R1, Phi-4, Llama-3.3-70B
8. **Use API-based models** (if allowed/budget permits):
   - Claude, GPT-4o, Gemini for higher accuracy
   - Could be used selectively on harder problems

### Phase 4: Advanced Techniques (High Effort, High Gain)
9. **Ensemble methods:**
   - Run multiple different models, aggregate answers
   - Weight models by per-category accuracy
10. **Fine-tuning:**
    - Fine-tune on public set math problems (competition rules allow it)
    - Use math-specific training data (MATH, GSM8K, etc.)
11. **Difficulty-adaptive pipeline:**
    - Classify question difficulty
    - Use cheaper/faster model for easy questions
    - Use stronger model + more samples for hard questions

---

## Key Technical Notes

- **Scoring edge cases:** The judger is sophisticated — it handles symbolic equivalence, interval unions, set operations, trigonometric identities, and text-to-number conversion. Your model doesn't need perfect LaTeX, but it MUST use `\boxed{}` for answer extraction.
- **MCQ extraction fallback:** If `\boxed{}` fails, the judger looks for the last uppercase letter — but relying on this is fragile.
- **Free-form multi-answer:** Questions with `[ANS]` placeholders expect comma-separated answers inside a single `\boxed{}`. The number of answers must match exactly.
- **Kaggle auth:** Uses `KGAT_` token format via `KAGGLE_API_TOKEN` env var, not the old `kaggle.json` method.

---

## Risk Assessment
- **Biggest risk:** Not submitting early. Without a baseline submission, you can't measure progress.
- **Common failure modes:** Answer formatting errors (missing `\boxed{}`), multi-answer count mismatch, symbolic form not recognized by judger.
- **Time management:** Phase 1-2 should take 1-2 days. Phase 3 takes a few days. Phase 4 is ongoing optimization over weeks.
