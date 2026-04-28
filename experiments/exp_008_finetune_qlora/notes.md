# Experiment: finetune_qlora

**Date:** 2026-04-27
**Baseline compared against:** exp_004_fewshot_prompts (local: 55.33%, Kaggle: pending)

## Hypothesis

QLoRA fine-tuning on Qwen3-4B using NuminaMath-CoT + self-generated correct responses from public.jsonl will push accuracy beyond the prompt-engineering ceiling (~55%). Prompt engineering is exhausted — every math few-shot attempted has regressed. Fine-tuning teaches the model to reliably conclude with `\boxed{}` (currently 23% missing_boxed) and improves accuracy on the weakest topics (trig 38.9%, number_theory 35.3%, differential_eq 25.0%) that prompting alone cannot reach.

## Change from baseline

- **Model weights:** QLoRA fine-tune of Qwen3-4B (r=16, alpha=32, 4-bit base) on Colab Pro A100 40GB
- **Training data:**
  1. NuminaMath-CoT (`AI4Math/NuminaMath-CoT` on HuggingFace) — ~25K filtered examples covering competition math (algebra, geometry, number theory, trig, calculus). Formatted to match our system prompt + `<think>...</think>` structure.
  2. Self-generated correct responses from public.jsonl — ~530 on-distribution examples (questions where exp_004 answered correctly).
- **Prompts:** exp_004 config unchanged (best MCQ few-shots, no math few-shots)
- **Inference:** Merged float16 weights pushed to private HuggingFace repo → pulled on Kaggle T4 x2 via huggingface_hub, loaded with vLLM as usual

## Notebook

**Training notebook (Colab Pro A100):**
https://colab.research.google.com/github/Trevis8688/151B_SP26_Competition/blob/exp/008_finetune_qlora/experiments/exp_008_finetune_qlora/train_colab.ipynb

### What the notebook does (cell by cell)

| Cell | What it does |
|------|-------------|
| 1 | Installs `unsloth`, `trl`, `datasets`, `bitsandbytes` |
| 2 | Config variables — reads your HF token from Colab Secrets (`HF_TOKEN`), sets repo name, training hyperparams (r=16, 2 epochs, lr=2e-4) |
| 3 | Loads Qwen3-4B in 4-bit quantization and wraps it with QLoRA adapters (only ~1% of params are trainable) |
| 4 | Defines the system prompts — identical to exp_004 (best config found so far) |
| 5 | Helper functions: `extract_boxed_and_reasoning()` splits a response into reasoning + final `\boxed{}`, and `make_messages()` formats it into the `<think>…</think>` structure Qwen3 expects |
| 6 | Downloads NuminaMath-CoT from HuggingFace (`AI-MO/NuminaMath-CoT`), filters to examples with `\boxed{}` in the solution, samples 25K, and formats each one as a training example |
| 7 | Uploads 3 local files (prompted one at a time), then loads the 623 correct responses from exp_004 and formats them as training examples using the same `<think>` structure |
| 8 | Combines both datasets (~25.6K total), shuffles, filters out examples longer than 2048 tokens, and builds a HuggingFace Dataset |
| 9 | Runs SFT training with TRL's `SFTTrainer` — 2 epochs, effective batch size 16, cosine LR decay, ~1.5–2 hrs on A100 |
| 10 | Merges the LoRA adapter back into the base model weights and saves as float16 (~8GB) — this is what vLLM will load |
| 11 | Pushes the merged model to a private HuggingFace repo for use on Kaggle |

## Plan & Progress

### Step 1 — Generate self-correct dataset from public.jsonl
- [ ] Run exp_004 config on full public.jsonl (or reuse existing scored results)
- [ ] Filter to correct responses → save as `data/sft_public_correct.jsonl`
- [ ] Format: `{"system": ..., "user": question, "assistant": "<think>...</think>\n\\boxed{answer}"}`

### Step 2 — Prepare NuminaMath-CoT subset
- [ ] Load `AI4Math/NuminaMath-CoT` from HuggingFace
- [ ] Filter: English only, remove trivial problems, cap at 25K examples
- [ ] Format to match our system prompt structure (free-form only — NuminaMath has no MCQ)
- [ ] Verify `\boxed{}` present in all targets

### Step 3 — Fine-tuning notebook (Colab Pro A100)
- [ ] Scaffold `experiments/exp_008_finetune_qlora/train_colab.ipynb`
- [ ] QLoRA: r=16, lora_alpha=32, target_modules = q/k/v/o_proj + gate/up/down_proj
- [ ] 2 epochs, batch_size=4, grad_accum=4 (effective batch 16), lr=2e-4 cosine decay
- [ ] Eval on held-out 100 public questions mid-training (track `\boxed{}` format rate)
- [ ] After training: merge adapter → save merged float16 weights

### Step 4 — Upload merged model
- [ ] Push merged Qwen3-4B weights to private HuggingFace repo
- [ ] Repo ID: ___________

### Step 5 — Kaggle inference
- [ ] Add HF token as Kaggle secret
- [ ] Update inference notebook to pull model from HF repo at runtime
- [ ] Run dev split (200q) first to validate
- [ ] If dev > 55.33%: run full private inference and submit

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
