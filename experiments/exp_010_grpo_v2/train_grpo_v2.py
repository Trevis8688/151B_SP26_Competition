"""GRPO v2 training script — DSMLP batch-mode friendly.

Ported from exp_009/train_grpo.ipynb. Changes from exp_009:
  - LR 2e-5 → 1e-5 (config.json)
  - BETA 0.01 → 0.02 (config.json)
  - curriculum: sweet_spot_ids.json (196) instead of sweet_spot_ids_clean.json (70)
  - epochs: 1 (aborted at 0.8) → 2 (full)
  - save: Drive callback removed; save_steps=25 to container local disk
  - HF push at end (replaces Cell 10/11 of exp_009 notebook)

Usage (DSMLP):
    K8S_TIMEOUT_SECONDS=43200 launch.sh -g 1 -m 32 -c 8 -B \\
        -- bash -c 'cd /home/$USER/151B_SP26_Competition && \\
                    pip install -q -r experiments/exp_010_grpo_v2/requirements.txt && \\
                    HF_TOKEN=$HF_TOKEN python experiments/exp_010_grpo_v2/train_grpo_v2.py'

Monitor:
    kubectl logs -f <pod_name>

Requires env var HF_TOKEN.
Expects repo layout:
    <repo>/data/public.jsonl
    <repo>/data/splits/dev.jsonl
    <repo>/judger.py
    <repo>/utils.py
    <repo>/experiments/exp_010_grpo_v2/{config.json, prompts.py, sweet_spot_ids.json}
"""
import json
import os
import sys
import shutil
import gc
import re
import random
from pathlib import Path
from unittest.mock import MagicMock

# ============================================================
# 0. Environment — set BEFORE importing torch / trl
# ============================================================
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("BNB_CUDA_VERSION", "124")  # bitsandbytes wheel match

# Mock vLLM and its tree of optional imports so TRL doesn't blow up trying to
# import them when use_vllm=False. Same fix as exp_009 Cell 8.
for mod in [
    "vllm", "vllm.sampling_params", "vllm.distributed",
    "vllm.distributed.device_communicators",
    "vllm.distributed.device_communicators.pynccl",
    "vllm.distributed.utils",
    "vllm_ascend", "vllm_ascend.distributed",
    "vllm_ascend.distributed.device_communicators",
    "vllm_ascend.distributed.device_communicators.pyhccl",
    "mergekit", "mergekit.config", "mergekit.merge",
    "llm_blender",
]:
    sys.modules[mod] = MagicMock()

# ============================================================
# 1. Paths + config
# ============================================================
EXP_DIR  = Path(__file__).resolve().parent
REPO_DIR = EXP_DIR.parent.parent

CONFIG     = json.loads((EXP_DIR / "config.json").read_text())
CURRICULUM = json.loads((EXP_DIR / "sweet_spot_ids.json").read_text())
SWEET_IDS  = set(CURRICULUM["sweet_ids"])

PUBLIC_JSONL = REPO_DIR / "data" / "public.jsonl"
DEV_JSONL    = REPO_DIR / "data" / "splits" / "dev.jsonl"

# Repo root on sys.path so we can import judger + utils + prompts
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(EXP_DIR))

print(f"REPO_DIR:       {REPO_DIR}")
print(f"EXP_DIR:        {EXP_DIR}")
print(f"Curriculum:     {len(SWEET_IDS)} prompts from sweet_spot_ids.json")
print(f"Output repo:    {CONFIG['output_repo']}")

BASE_MODEL = CONFIG["base_model"]
TRAIN      = CONFIG["training"]
OUTPUT_REPO = CONFIG["output_repo"]

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("WARN: HF_TOKEN env var not set — training will run but final push will fail.")

# ============================================================
# 2. Imports (torch must come after env vars; trl after MagicMock)
# ============================================================
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import HfApi

# Project-local modules
from judger import Judger  # noqa: E402
from prompts import SYSTEM_PROMPT_MATH, SYSTEM_PROMPT_MCQ, FEWSHOT_MATH, FEWSHOT_MCQ  # noqa: E402

# bf16 only on Ampere+ (SM 8.0+). On Volta/Turing (V100, 2080Ti, T4) fall back to fp16.
BF16_OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if BF16_OK else torch.float16
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"bf16 supported: {BF16_OK}  → compute dtype: {COMPUTE_DTYPE}")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================
# 3. Tokenizer + 4-bit base + LoRA wrap
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)

print(f"\nLoading tokenizer + 4-bit model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=COMPUTE_DTYPE,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=TRAIN["lora_r"],
    lora_alpha=TRAIN["lora_alpha"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=TRAIN["lora_dropout"],  # 0.0 — non-zero corrupts rollouts under PEFT
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# 4. Build dataset: public.jsonl ∩ SWEET_IDS, excl. dev IDs
# ============================================================
LETTERS = "ABCDEFGHIJ"
JUDGER = Judger()

dev_ids = set()
with open(DEV_JSONL) as f:
    for line in f:
        dev_ids.add(json.loads(line)["id"])
print(f"\nDev IDs excluded: {len(dev_ids)}")

rows = []
with open(PUBLIC_JSONL) as f:
    for line in f:
        ex = json.loads(line)
        if ex["id"] in dev_ids:
            continue
        if ex["id"] not in SWEET_IDS:
            continue
        is_mcq = bool("options" in ex and ex["options"])
        question_text = ex["question"]
        if is_mcq:
            opts = "\n".join(f"{LETTERS[i]}. {v}" for i, v in enumerate(ex["options"]))
            question_text = f"{question_text}\n\nOptions:\n{opts}"
            sys_prompt = SYSTEM_PROMPT_MCQ
            fewshots = FEWSHOT_MCQ
        else:
            sys_prompt = SYSTEM_PROMPT_MATH
            fewshots = FEWSHOT_MATH

        msgs = [{"role": "system", "content": sys_prompt}]
        msgs.extend(fewshots)
        msgs.append({"role": "user", "content": question_text})

        rows.append({
            "prompt":       msgs,
            "answer_json":  json.dumps(ex["answer"]),
            "options_json": json.dumps(ex.get("options", [])),
            "is_mcq":       is_mcq,
            "id":           ex["id"],
        })

random.shuffle(rows)
train_dataset = Dataset.from_list(rows)
print(f"Train set: {len(rows)} prompts  (MCQ: {sum(r['is_mcq'] for r in rows)}, "
      f"FF: {sum(not r['is_mcq'] for r in rows)})")

# ============================================================
# 5. Reward functions (correctness + granular format)
# ============================================================
_BOXED_RE = re.compile(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")

def extract_post_think(text: str) -> str:
    idx = text.rfind("</think>")
    return text if idx == -1 else text[idx + len("</think>"):]

def correctness_reward(prompts, completions, answer_json=None, options_json=None,
                       is_mcq=None, **kwargs):
    rewards = []
    for i, comp in enumerate(completions):
        text = "".join(m.get("content", "") for m in comp) if isinstance(comp, list) else str(comp)
        post = extract_post_think(text)
        gold = json.loads(answer_json[i]) if answer_json else []
        opts = json.loads(options_json[i]) if options_json else []
        try:
            opts_list = ([opts] * len(gold)) if gold else [None]
            ok = JUDGER.auto_judge(post, gold, opts_list)
        except Exception:
            ok = False
        rewards.append(1.0 if ok else 0.0)
    return rewards

def format_reward(prompts, completions, **kwargs):
    rewards = []
    for comp in completions:
        text = "".join(m.get("content", "") for m in comp) if isinstance(comp, list) else str(comp)
        r = 0.0
        has_close_think = "</think>" in text
        if has_close_think:
            r += 0.05
        all_boxed = _BOXED_RE.findall(text)
        if all_boxed:
            r += 0.10
        post = extract_post_think(text)
        post_boxed = _BOXED_RE.findall(post)
        if has_close_think and post_boxed:
            r += 0.05
        if len(post_boxed) == 1:
            r += 0.025
        rewards.append(r)
    return rewards

# Self-test
_t = "<think>let me reason \\boxed{42}</think>\nFinal: \\boxed{42}"
assert format_reward([], [_t])[0] == 0.225
assert format_reward([], ["<think>...</think>\n\\boxed{1} \\boxed{2}"])[0] == 0.20
print("Reward self-test OK.\n")

# ============================================================
# 6. GRPOConfig + monkey-patches
# ============================================================
CKPT_DIR    = EXP_DIR / "checkpoints"
ADAPTER_DIR = EXP_DIR / "adapter_final"
MERGED_DIR  = EXP_DIR / "merged_final"
if CKPT_DIR.exists():
    shutil.rmtree(CKPT_DIR)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

training_args = GRPOConfig(
    output_dir=str(CKPT_DIR),
    num_train_epochs=TRAIN["epochs"],
    per_device_train_batch_size=TRAIN["per_device_train_batch_size"],
    gradient_accumulation_steps=TRAIN["gradient_accumulation_steps"],
    learning_rate=TRAIN["learning_rate"],
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="adamw_8bit",
    bf16=BF16_OK,
    fp16=not BF16_OK,
    logging_steps=1,
    save_strategy=TRAIN["save_strategy"],
    save_steps=TRAIN["save_steps"],
    save_total_limit=TRAIN["save_total_limit"],
    seed=RANDOM_SEED,
    report_to="none",
    # GRPO-specific
    num_generations=TRAIN["num_generations"],
    max_prompt_length=TRAIN["max_prompt_length"],
    max_completion_length=TRAIN["max_completion_length"],
    temperature=TRAIN["temperature_train"],
    beta=TRAIN["beta"],
    use_vllm=False,
)

# Restore original generate, then wrap with eval()/train() flip so use_cache is
# respected during rollouts (5× speedup). Same fix as exp_009 Cell 8.
if hasattr(type(model), "generate"):
    model.generate = type(model).generate.__get__(model)
_orig_generate = model.generate
def _eval_generate(*args, **kwargs):
    model.eval()
    out = _orig_generate(*args, **kwargs)
    model.train()
    return out
model.generate = _eval_generate

class RewardLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        keys = ("loss", "reward", "kl", "completion")
        relevant = {k: v for k, v in logs.items() if any(s in k for s in keys)}
        if relevant:
            parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in relevant.items()]
            print(f"  [step {state.global_step}] " + "  ".join(parts), flush=True)

if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

# ============================================================
# 7. Train
# ============================================================
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward, format_reward],
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[RewardLogCallback()],
)

steps_per_epoch = len(train_dataset) // (TRAIN["per_device_train_batch_size"]
                                         * TRAIN["gradient_accumulation_steps"])
total_steps = steps_per_epoch * TRAIN["epochs"]
tokens_per_epoch = (len(train_dataset) * TRAIN["num_generations"]
                    * TRAIN["max_completion_length"]) / 1e6

print("\n" + "=" * 60)
print("Starting GRPO training")
print(f"  Epochs:           {TRAIN['epochs']}")
print(f"  Steps/epoch:      ~{steps_per_epoch}")
print(f"  Total steps:      ~{total_steps}")
print(f"  Tokens/epoch:     ~{tokens_per_epoch:.1f}M")
print(f"  LR:               {TRAIN['learning_rate']}")
print(f"  BETA:             {TRAIN['beta']}")
print(f"  num_generations:  {TRAIN['num_generations']}")
print(f"  max_completion:   {TRAIN['max_completion_length']}")
print("=" * 60 + "\n", flush=True)

trainer.train()

# Save final adapter
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print(f"\nFinal adapter saved: {ADAPTER_DIR}")

# ============================================================
# 8. Merge LoRA → fp16/bf16 base → push to HF
# ============================================================
print("\nFreeing training memory before fp16 merge ...")
del trainer
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU free before merge: {free/1e9:.1f}/{total/1e9:.1f} GB")

MERGE_DTYPE = torch.float16  # safer for downstream Kaggle T4 inference (no bf16 on T4)
print(f"\nLoading {BASE_MODEL} in {MERGE_DTYPE} for merge ...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=MERGE_DTYPE,
    device_map="auto",
    trust_remote_code=True,
)
peft_model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
merged = peft_model.merge_and_unload()
merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
tokenizer.save_pretrained(str(MERGED_DIR))
print(f"Merged model saved: {MERGED_DIR}")

if HF_TOKEN:
    print(f"\nPushing to https://huggingface.co/{OUTPUT_REPO} ...")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(MERGED_DIR),
        repo_id=OUTPUT_REPO,
        repo_type="model",
        token=HF_TOKEN,
    )
    print(f"Done. Set Kaggle inference config.model_id = '{OUTPUT_REPO}'")
else:
    print("\nSkipped HF push (HF_TOKEN not set). Merged model is on local disk only.")

print("\nALL DONE.")
