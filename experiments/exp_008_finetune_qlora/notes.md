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

### Key hyperparams (Cell 2)

| Param | Value | Rationale |
|-------|-------|-----------|
| MAX_SEQ_LENGTH | 1024 | Halved from 2048 — attention is O(n²); biggest single speedup |
| NUMINAMATH_N | 5000 | Fits A100 compute budget (~25-75 min) |
| TRAIN_BATCH_SIZE | 16 | No grad accumulation needed on 80GB — eliminates accum overhead |
| GRAD_ACCUM_STEPS | 1 | Effective batch = 16 |
| NUM_EPOCHS | 2 | 5K × 2 gives LoRA enough steps to converge on `<think>` format |
| LEARNING_RATE | 2e-4 | Standard for QLoRA |

### What the notebook does (cell by cell)

| Cell | What it does |
|------|-------------|
| 1 | Installs `unsloth`, `trl`, `datasets`, `bitsandbytes` |
| 2 | Config variables — reads your HF token from Colab Secrets (`HF_TOKEN`), sets repo name, training hyperparams |
| 3 | Loads Qwen3-4B in 4-bit quantization and wraps it with QLoRA adapters. `use_gradient_checkpointing=False` — 80GB A100 has VRAM headroom, avoids ~35% throughput tax |
| 4 | Defines the system prompts — identical to exp_004 (best config found so far) |
| 5 | Helper functions: `extract_boxed_and_reasoning()` splits a response into reasoning + final `\boxed{}`, and `make_messages()` formats it into the `<think>…</think>` structure Qwen3 expects |
| 6 | Downloads NuminaMath-CoT from HuggingFace (`AI-MO/NuminaMath-CoT`), filters to examples with `\boxed{}` in the solution, samples 5K, and formats each one as a training example |
| 7 | Uploads 3 local files (prompted one at a time), then loads the 623 correct responses from exp_004 and formats them as training examples using the same `<think>` structure |
| 8 | Combines both datasets (~5.6K total), shuffles, filters out examples longer than 1024 tokens, and builds a HuggingFace Dataset |
| 8b | **Format sanity check** — prints a rendered training example alongside the inference-time prompt. Must visually confirm assistant continuation looks correct before burning compute |
| 9 | Splits off 200 eval examples, then runs SFT training — 2 epochs, batch=16, bf16, cosine LR, eval every 50 steps to catch format bugs mid-training |
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

## Proposed next steps (further fine-tuning)

Yes, more fine-tuning can be done after this run. The merged float16 model on HF and the saved LoRA adapter checkpoint (`/content/exp008_checkpoints/checkpoint-*/`) are both starting points. Listed in rough priority order:

### 1. Self-distillation round 2 (highest expected gain)
Run inference with the exp_008 model on the **full 1126-question public.jsonl**. Filter to correct responses — should be a substantially larger pool than the 623 from exp_004 (e.g., if exp_008 hits 60% local, ~675 examples; if 65%, ~730). Retrain a fresh QLoRA on `NuminaMath_5K + public_correct_v2`. This is the STaR / ReST loop — proven to compound across rounds for math reasoning.

- New experiment slug: `exp_009_selfdistill_v2`
- Same notebook structure, swap the public_responses file
- Stop iterating when correct-pool size plateaus

### 2. Topic-targeted oversampling
Run `scripts/analyze.py` on exp_008 dev results. Whichever topics still fail (likely trig / number_theory / differential_eq based on exp_004 baseline), filter NuminaMath-CoT to those topics specifically and oversample (5K topic-targeted + 5K mixed).

### 3. Continue training the existing adapter
Cheaper than starting over if eval loss was still decreasing at end of run. Reload the last checkpoint with `FastLanguageModel.from_pretrained(checkpoint_path)`, train another 1-2 epochs on the same data — or on a fresh NuminaMath sample.

### 4. Scale NuminaMath subset (15K-25K)
If round 1 finished comfortably under the compute budget (e.g., < 45 min), scale up. With packing + seq=1024 + grad_checkpointing=False, 25K should be feasible in 2-3 hrs. Pair with rank bump to r=32 if the data scales up.

### 5. DPO / preference fine-tuning
Use `(correct_response, incorrect_response)` pairs from public.jsonl as a preference dataset. TRL's `DPOTrainer` runs on top of the SFT'd model. This nudges reasoning *style* (e.g., "always close with `\boxed{}`") rather than teaching new skills — well-suited if exp_008 has high knowledge but inconsistent format.

### 6. Diversify data sources
If gains plateau on NuminaMath: MetaMathQA, OpenMathInstruct-2, GSM8K, MATH train split. Mix at 70% in-distribution (NuminaMath + public correct) / 30% diversification to avoid catastrophic shift away from competition format.

### Decision rule
After exp_008 dev results land:
- If dev >> 55.33% (e.g., 62%+): go to **(1) self-distillation** — bigger correct pool, compound the gain
- If dev ≈ baseline but `\boxed{}` format rate improved: go to **(3) continue training** — undertrained on skills
- If specific topics regressed: go to **(2) topic-targeted oversampling**
- If stuck or noisy: go to **(5) DPO** to clean up format/style
