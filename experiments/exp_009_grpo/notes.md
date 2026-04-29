# Experiment: grpo

**Date:** 2026-04-28
**Baseline compared against:** exp_004_fewshot_prompts (local: 55.33%, Kaggle: pending)

## Hypothesis

GRPO fine-tuning on Qwen3-4B-Thinking-2507 using public.jsonl as reward signal will push accuracy beyond the prompt-engineering ceiling (~55%) by training the model to discover correct reasoning paths — unlike SFT (which only imitates already-correct solutions), GRPO generates multiple responses per question and reinforces the ones that produce correct `\boxed{}` answers, allowing it to improve on problems the base model currently fails.

**Key risk:** Sparse reward on the hardest topics (differential_eq 16.7%, number_theory 29.4%) — with a weak base policy, most generation batches will be uniformly wrong, producing near-zero gradient signal exactly where improvement is needed. Mitigation: 8 generations per prompt (more variance), a format bonus reward (partial credit for `\boxed{}` presence to ensure gradients even on wrong answers), and starting from the clean base model.

## Change from baseline

- **Model:** Qwen/Qwen3-4B-Thinking-2507 base (NOT exp_008 checkpoint — that has MCQ damage)
- **Training method:** GRPO via unsloth's GRPOTrainer (memory efficient on A100)
- **Training data:** `data/public.jsonl` minus `data/splits/dev.jsonl` (~926 questions, all topics)
- **Reward function:**
  - Correctness reward: +1.0 if `Judger.auto_judge()` passes, 0.0 otherwise
  - Format bonus: +0.1 if response contains `\boxed{}` (ensures gradient on uniformly-wrong batches)
- **num_generations:** 8 per prompt (better signal density on hard topics vs 4)
- **max_completion_length:** 4096 (must see full `<think>...</think>` chain + answer)
- **Prompts:** exp_004 config unchanged (best MCQ few-shots, no math few-shots)
- **Success metric:** beat 55.33% on dev split

## Notebook

**Training notebook (Colab Pro A100 80GB required):**
https://colab.research.google.com/github/Trevis8688/151B_SP26_Competition/blob/exp/009_grpo/experiments/exp_009_grpo/train_grpo.ipynb

### Pilot first
Cell 2 has `PILOT_MODE = True` → runs on 50 prompts, ~30 min. **Verify the logged reward trends upward before flipping to full mode.** If reward is flat after the pilot, the reward signal is too sparse (mean ~0/8 per batch) and we need to either:
- Pre-filter training prompts to ones where base model gets 1–7 of 8 correct (sweet-spot for GRPO)
- Increase the format bonus weight
- Lower NUM_GENERATIONS to fit a curriculum approach

### Hyperparameters

| Param | Value | Rationale |
|-------|-------|-----------|
| MODEL_NAME | Qwen3-4B-Thinking-2507 | Competition mandate; exp_008 mistakenly used non-thinking variant |
| NUM_GENERATIONS | 8 | Sparse-reward mitigation; better signal density on hard topics |
| MAX_COMPLETION_LEN | 4096 | Must fit full `<think>...</think>` chain + answer |
| MAX_PROMPT_LENGTH | 1024 | Prompt p99 measured ≈ 851 tokens |
| LEARNING_RATE | 5e-6 | RL is unstable at SFT-scale LRs |
| BETA (KL) | 0.04 | Standard; constrains drift from base policy |
| TEMPERATURE | 1.0 (train) / 0.6 (eval) | Higher exploration during training |
| LORA_R / ALPHA | 16 / 32 | Same as exp_008 — adequate capacity |
| use_gradient_checkpointing | "unsloth" | Required — 4096-token completions OOM otherwise |
| Effective prompt batch | 4 | per_device=1 × accum=4 (each prompt → 8 generations = 32 sequences/step) |

### Reward signals
- **Correctness:** +1.0 if `Judger.auto_judge(post_think_text, gold, options)` passes, 0.0 otherwise
- **Format bonus:** +0.1 if `\boxed{}` appears AFTER `</think>` (ignores boxed inside reasoning)

### Critical: parse after `</think>`
Qwen3-Thinking emits `<think>...</think>` then the answer. Reward extraction MUST use `extract_post_think()` — boxed answers inside the thinking block must not be rewarded, or the model learns to short-circuit reasoning.

## Dev results

_Fill in after running analyze.py on results.jsonl (split=dev)._

| Metric | Baseline (exp_004) | This (dev) | Δ |
|--------|-------------------:|-----------:|---:|
| Overall | 55.33% | | |
| MCQ | 63.20% | | |
| Free-form | 51.40% | | |

## Topic movers

_Top 3 topics that improved / regressed._

## Conclusion

- [ ] Keep (merge into `main` prompt set)
- [ ] Discard
- [ ] Needs variant — next experiment idea:
