"""GRPO v2 training script — DSMLP batch-mode friendly.

Convert of exp_009/train_grpo.ipynb to a runnable .py for `launch.sh -B`.

Usage (DSMLP):
    K8S_TIMEOUT_SECONDS=43200 launch.sh -g 1 -v <gpu_type> -m 32 -c 8 -B \
        -i trevorduong/grpo-v2:latest \
        -- python /opt/exp_010/train_grpo_v2.py

Monitor:
    kubectl logs -f <pod_name>

Status: SKELETON — most of the body is TODO blocks lifted from train_grpo.ipynb.
Before running, port the working cells from exp_009/train_grpo.ipynb and replace
the placeholders below. See the Gemini MagicMock patch in Cell 8 — that goes here.

Key changes from exp_009:
- LR 2e-5 → 1e-5
- BETA 0.01 → 0.02
- curriculum: sweet_spot_ids.json (196) instead of sweet_spot_ids_clean.json (70)
- epochs: 1 (aborted at 0.8) → 2 (full)
- save: Drive callback → save_steps=25 to container local disk
"""
import json
import os
import sys
from pathlib import Path

# ============================================================
# 0. Environment — set BEFORE importing torch / trl / vllm
# ============================================================
# DSMLP doesn't need the Colab stdout patch (no ipykernel),
# but VLLM_USE_V1=0 is still useful while we test the install.
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ============================================================
# 1. Config
# ============================================================
EXP_DIR = Path(__file__).parent
CONFIG = json.loads((EXP_DIR / "config.json").read_text())
CURRICULUM = json.loads((EXP_DIR / "sweet_spot_ids.json").read_text())
SWEET_IDS = set(CURRICULUM["sweet_ids"])
print(f"Loaded curriculum: {len(SWEET_IDS)} sweet-spot prompts")
print(f"Distribution: {CURRICULUM.get('distribution', 'N/A')}")

BASE_MODEL = CONFIG["base_model"]
OUTPUT_REPO = CONFIG["output_repo"]
TRAIN = CONFIG["training"]
REWARD = CONFIG["reward"]

# ============================================================
# 2. Imports (torch, trl, peft, datasets)
# ============================================================
# TODO: copy from exp_009/train_grpo.ipynb Cell 2
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model, PeftModel
# from trl import GRPOConfig, GRPOTrainer
# from datasets import Dataset
# import sys
# from unittest.mock import MagicMock
# for mod in ['vllm', 'vllm.utils', 'vllm.engine', 'vllm.engine.arg_utils']:
#     sys.modules[mod] = MagicMock()  # Gemini patch — TRL imports vllm even when unused

# ============================================================
# 3. Tokenizer + 4-bit base model
# ============================================================
# TODO: copy from exp_009 Cell 3

# ============================================================
# 4. LoRA wrap + monkey-patch generate() with eval()/train() brackets
# ============================================================
# TODO: copy from exp_009 Cell 4
# IMPORTANT: lora_dropout MUST be 0.0 — non-zero corrupts rollouts under TRL+PEFT

# ============================================================
# 5. Load + filter dataset (public.jsonl ∩ SWEET_IDS)
# ============================================================
# TODO: copy from exp_009 Cell 5
# Apply system prompt + 3 MCQ few-shots from prompts.py

# ============================================================
# 6. Reward function (correctness + granular format, post-think parse)
# ============================================================
# TODO: copy from exp_009 Cell 6
# Must use extract_post_think() to avoid rewarding boxed answers inside <think>

# ============================================================
# 7. GRPOConfig
# ============================================================
# config = GRPOConfig(
#     output_dir="./checkpoints",
#     num_train_epochs=TRAIN["epochs"],
#     learning_rate=TRAIN["learning_rate"],
#     beta=TRAIN["beta"],
#     num_generations=TRAIN["num_generations"],
#     max_prompt_length=TRAIN["max_prompt_length"],
#     max_completion_length=TRAIN["max_completion_length"],
#     per_device_train_batch_size=TRAIN["per_device_train_batch_size"],
#     gradient_accumulation_steps=TRAIN["gradient_accumulation_steps"],
#     temperature=TRAIN["temperature_train"],
#     save_strategy=TRAIN["save_strategy"],
#     save_steps=TRAIN["save_steps"],
#     save_total_limit=TRAIN["save_total_limit"],
#     logging_steps=1,
#     bf16=True,        # DSMLP GPUs likely support bf16 (Ampere/Hopper)
#     report_to=[],     # no wandb on DSMLP
# )

# ============================================================
# 8. Train
# ============================================================
# trainer = GRPOTrainer(
#     model=peft_model,
#     reward_funcs=[reward_fn],
#     args=config,
#     train_dataset=ds,
#     processing_class=tokenizer,
# )
# trainer.train()

# ============================================================
# 9. Merge LoRA + push to HF Hub
# ============================================================
# merged = peft_model.merge_and_unload()
# merged.push_to_hub(OUTPUT_REPO, private=False)
# tokenizer.push_to_hub(OUTPUT_REPO)
# print(f"Pushed merged model to {OUTPUT_REPO}")

if __name__ == "__main__":
    print("SKELETON ONLY — port the train_grpo.ipynb cells into this script before running.")
    print(f"Target output: {OUTPUT_REPO}")
    sys.exit(1)
