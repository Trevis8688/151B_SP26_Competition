# CSE 151B Spring 2026 — LLM Mathematical Reasoning

Our submission answers the competition's math problems with a **two-stage GRPO
pipeline** built on `Qwen3-4B-Thinking-2507`. The entire pipeline — model
loading, inference on the private set, and all post-processing — runs from a
**single function**, `run_inference()` in [`run_inference.py`](run_inference.py).

Private leaderboard score: **0.660** (updated judge).

---

## Quickstart

```bash
# 1. Install dependencies (Python 3.10+, CUDA GPU required)
pip install -r requirements.txt

# 2. (Only if the model repos are private) authenticate to the HuggingFace Hub
huggingface-cli login

# 3. Run the full pipeline → writes submission.csv
python run_inference.py --private-path data/private.jsonl --output-csv submission.csv
```

Or call it from Python:

```python
from run_inference import run_inference
run_inference(private_path="data/private.jsonl", output_csv="submission.csv")
```

That single call loads both models, runs both stages, applies the rescue
post-processing, and writes the final `submission.csv` (columns `id,response`,
one row per private question). Nothing manual or external is required.

---

## Hardware & runtime

| | |
|---|---|
| **GPU used for the submission** | Kaggle **T4 × 2** (16 GB each, SM 7.5) |
| **Precision** | float16 (the T4 has no bfloat16) |
| **Approx. inference time** | ~80 min for the full public+private run on T4×2; **private-only (943 questions) ≈ 45–60 min** |

`run_inference.py` defaults vLLM tensor parallelism to **1 GPU** — the
configuration the submission's stage-1 ran under, and the only value safe on
every GPU (`tensor_parallel_size=2` has a known CUDA-graph crash on the T4).
Pass `--tensor-parallel-size 2` on Ampere+ multi-GPU hardware for speed. The two
4B models (~8 GB each in fp16) run in **separate subprocesses** (see below), so
only one is resident at a time — a single 16 GB GPU is sufficient.

### Process isolation (why two stages, one function)

The pipeline uses two different models. To switch models cleanly without
in-process vLLM teardown (version-fragile, and prone to OOM the second model at
`gpu_memory_utilization=0.90`), `run_inference()` launches **each stage as a
fresh subprocess of `run_inference.py`**. On stage exit the OS reclaims all of
that stage's VRAM, guaranteed. The orchestrator process never touches the GPU —
it only spawns the workers and assembles the final CSV. This is all internal:
you still call the single `run_inference()` function.

---

## Model weights

No manual download or directory placement is needed — both models load
automatically from the HuggingFace Hub when `run_inference()` runs:

| Stage | HuggingFace repo | Role |
|---|---|---|
| Stage 1 (generation) | [`TrevorDuong/qwen3-4b-thinking-grpo-pass2`](https://huggingface.co/TrevorDuong/qwen3-4b-thinking-grpo-pass2) | GRPO pass-2 policy, merged (single safetensors) |
| Stage 2 (rescue)     | [`TrevorDuong/qwen3-4b-thinking-grpo-strict70`](https://huggingface.co/TrevorDuong/qwen3-4b-thinking-grpo-strict70) | GRPO pass-1 policy, used as answer-extractor |

Both are full merged checkpoints fine-tuned from `Qwen/Qwen3-4B-Thinking-2507`
with GRPO (reinforcement learning) on the public training problems — no LoRA
adapter switching at inference time. If the repos are private, run
`huggingface-cli login` (or set the `HF_TOKEN` env var) before running.

---

## How the pipeline works

`run_inference()` performs two stages; the second only touches responses the
first failed to finish.

**Stage 1 — generation** (`TrevorDuong/qwen3-4b-thinking-grpo-pass2`)
- Prompt: a system instruction + **3 MCQ few-shot examples** (MCQ questions
  only) + the question. The few-shots are required — the GRPO policy was trained
  with them, so removing them puts the model out-of-distribution.
- Sampling: `temperature=0.6, top_p=0.95, top_k=20, max_tokens=8192`.

**Stage 2 — rescue** (`TrevorDuong/qwen3-4b-thinking-grpo-strict70`)
- Targets only responses that never emitted a `\boxed{}` (truncated reasoning
  traces — ~20% of stage-1 outputs).
- Feeds the last 3000 tokens of the truncated trace plus the original question
  to an "answer-extractor" prompt and asks for just the final `\boxed{}`.
- Sampling: `temperature=0.1, top_p=0.95, top_k=20, max_tokens=4096`.
- When the rescue produces a `\boxed{}`, it is appended to the original response
  under a `[RESCUE EXTRACTION]` marker (additive — the original trace is kept).

All hyperparameters above are the final ones used for the submission and are
defined as constants at the top of `run_inference.py`.

---

## Reproducibility note

Outputs are sampled (`temperature > 0` in stage 1), so exact strings will vary
run to run; overall accuracy is stable. The pipeline corresponds to experiment
`exp_018_pass2_rescue` (see `experiments/` for the full experiment log and the
training scripts for both GRPO checkpoints).

---

## Repository layout (key files)

| File | Description |
|---|---|
| `run_inference.py` | **Single entry point** — full pipeline → `submission.csv` |
| `judger.py` / `utils.py` | SymPy-based local scoring (free-form equivalence) |
| `data/public.jsonl` | Public set with ground-truth answers (local scoring) |
| `data/private.jsonl` | Private leaderboard test set (no answers) |
| `experiments/` | Per-experiment configs, prompts, notes, and training scripts |
| `scripts/score.py` | Score a `submission.csv` / responses against `public.jsonl` |
