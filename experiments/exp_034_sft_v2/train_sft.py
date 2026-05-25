"""
exp_034 — QLoRA SFT trainer for qwen3-4b-thinking-grpo-pass2.

Two phases driven by --phase:
  probe : trains for cfg.probe_n examples (~30-32 steps at gas=16) and stops.
          Adapter saved to /tmp/sft_ckpt-probe. Eval script then runs the
          dev-probe gate; launcher decides whether to proceed.
  full  : resumes from probe adapter, trains the full train.jsonl to epoch end,
          merges + pushes to HF Hub as cfg.output_merged_repo.

HF Hub checkpoint callback pushes adapter+optim+sched every save_steps so a 12h
DSMLP pod cap can be resumed in a second pod (_try_resume_from_hf at startup).

Run inside the DSMLP launch_sft_v2.sh venv. Reads HF_TOKEN from env.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


# ============================================================
# 0. Args + config
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=["probe", "full"], required=True)
ap.add_argument("--config", default=str(Path(__file__).parent / "config.json"))
ap.add_argument("--data_dir", default="/tmp/sft_data")
ap.add_argument("--ckpt_root", default="/tmp")
args = ap.parse_args()

CONFIG = json.loads(Path(args.config).read_text())
BASE_MODEL = CONFIG["base_model_id"]
ADAPTER_REPO = CONFIG["output_adapter_repo"]
MERGED_REPO = CONFIG["output_merged_repo"]
DATA = CONFIG["data"]
LORA = CONFIG["lora"]
TRAIN = CONFIG["train"]

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("WARN: HF_TOKEN not set — no checkpoint pushing or final merge upload.")

CKPT_DIR = Path(args.ckpt_root) / f"sft_ckpt-{args.phase}"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = Path(args.ckpt_root) / f"sft_adapter-{args.phase}"
MERGED_DIR = Path(args.ckpt_root) / f"sft_merged-{args.phase}"

# Adapter checkpoint repo is phase-suffixed so probe and full don't collide on Hub.
ADAPTER_CKPT_REPO = f"{ADAPTER_REPO}-{args.phase}"

# Data file is the same for both phases; probe trains for max_steps and stops,
# full picks up the probe adapter and continues over the full file.
DATA_FILE = Path(args.data_dir) / "train.jsonl"
if not DATA_FILE.exists():
    raise SystemExit(f"Missing {DATA_FILE} — run prepare_data.py first.")


# ============================================================
# 1. Tokenizer + base model (4-bit)
# ============================================================
print(f"[load] tokenizer {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[load] base model {BASE_MODEL} in 4-bit")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # required for gradient checkpointing


# ============================================================
# 2. Data — apply chat template, completion-only loss
# ============================================================
print(f"[data] load {DATA_FILE}")
ds = load_dataset("json", data_files=str(DATA_FILE), split="train")
print(f"[data] {len(ds)} examples")


def format_for_chat(example):
    """Render messages through the tokenizer's chat template. Returns the full
    prompt+response text; the completion-only collator masks loss on everything
    before the response template."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


# Qwen3 chat template starts the assistant turn with `<|im_start|>assistant\n`.
# DataCollatorForCompletionOnlyLM masks all tokens up to and including this template,
# so the loss is computed only on the assistant content (including the </think> wrapper).
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=RESPONSE_TEMPLATE,
    tokenizer=tokenizer,
)


# ============================================================
# 3. SFT config
# ============================================================
# Probe stops at probe_n examples. With gas=16 and probe_n=500, that's ~31 steps.
probe_max_steps = max(1, DATA["probe_n"] // (TRAIN["per_device_train_batch_size"]
                                              * TRAIN["gradient_accumulation_steps"]))
max_steps = probe_max_steps if args.phase == "probe" else -1
num_train_epochs = 0 if args.phase == "probe" else TRAIN["num_train_epochs"]

sft_args = SFTConfig(
    output_dir=str(CKPT_DIR),
    per_device_train_batch_size=TRAIN["per_device_train_batch_size"],
    gradient_accumulation_steps=TRAIN["gradient_accumulation_steps"],
    num_train_epochs=num_train_epochs,
    max_steps=max_steps,
    learning_rate=TRAIN["learning_rate"],
    lr_scheduler_type=TRAIN["lr_scheduler_type"],
    warmup_ratio=TRAIN["warmup_ratio"],
    weight_decay=TRAIN["weight_decay"],
    max_grad_norm=TRAIN["max_grad_norm"],
    save_steps=TRAIN["save_steps"],
    save_strategy="steps",
    logging_steps=TRAIN["logging_steps"],
    bf16=TRAIN["bf16"],
    fp16=TRAIN["fp16"],
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_seq_length=DATA["max_seq_len"],
    packing=False,
    report_to="none",
    remove_unused_columns=False,
)

lora_cfg = LoraConfig(
    r=LORA["r"],
    lora_alpha=LORA["alpha"],
    lora_dropout=LORA["dropout"],
    target_modules=LORA["target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)


# ============================================================
# 4. Callbacks — log + HF-Hub push on save
# ============================================================
class LossLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        print(f"  [step {state.global_step}] loss={logs['loss']:.4f}"
              f"  lr={logs.get('learning_rate', 0):.2e}",
              flush=True)


class HFPushAdapterCallback(TrainerCallback):
    """Push trainer checkpoint to HF Hub on each save — survives 12h pod cap."""
    def __init__(self, hf_token, ckpt_repo):
        self.api = HfApi(token=hf_token)
        self.repo = ckpt_repo
        try:
            self.api.create_repo(ckpt_repo, private=True, exist_ok=True)
            print(f"[hf] checkpoint repo ready: {ckpt_repo}", flush=True)
        except Exception as e:
            print(f"WARN: create_repo({ckpt_repo}) failed: {e}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.exists():
            return
        try:
            self.api.upload_folder(
                folder_path=str(ckpt_dir),
                path_in_repo=f"checkpoint-{state.global_step}",
                repo_id=self.repo,
                ignore_patterns=["*.bin.tmp", "global_step*"],
            )
            print(f"  ↑ pushed checkpoint-{state.global_step} → "
                  f"https://huggingface.co/{self.repo}/tree/main/checkpoint-{state.global_step}",
                  flush=True)
        except Exception as e:
            print(f"  WARN: HF push failed for checkpoint-{state.global_step}: {e}", flush=True)


def _try_resume_from_hf(ckpt_repo, hf_token, local_ckpt_dir):
    if os.environ.get("DISABLE_RESUME"):
        return None
    if not ckpt_repo or not hf_token:
        return None
    api = HfApi(token=hf_token)
    try:
        files = api.list_repo_files(repo_id=ckpt_repo)
    except Exception as e:
        print(f"[resume] no prior ckpts at {ckpt_repo}: {e}")
        return None
    steps = set()
    for f in files:
        if f.startswith("checkpoint-"):
            try:
                steps.add(int(f.split("/", 1)[0].split("-", 1)[1]))
            except (ValueError, IndexError):
                pass
    if not steps:
        return None
    latest = max(steps)
    target = Path(local_ckpt_dir) / f"checkpoint-{latest}"
    print(f"[resume] downloading checkpoint-{latest} from {ckpt_repo}")
    snapshot_download(
        repo_id=ckpt_repo,
        allow_patterns=[f"checkpoint-{latest}/*"],
        local_dir=str(local_ckpt_dir),
        token=hf_token,
    )
    return str(target)


callbacks = [LossLogCallback()]
if HF_TOKEN:
    callbacks.append(HFPushAdapterCallback(HF_TOKEN, ADAPTER_CKPT_REPO))


def _find_probe_seed():
    """Locate probe adapter weights to seed a FRESH full start.
    Prefer the local /tmp dir (same-pod chaining); fall back to the -probe HF
    repo (relaunch in a new pod where /tmp was wiped).
    Returns a path to a peft adapter dir, or raises if none found."""
    local_probe = Path(args.ckpt_root) / "sft_adapter-probe"
    if (local_probe / "adapter_config.json").exists():
        print(f"[full] seeding from LOCAL probe adapter: {local_probe}")
        return str(local_probe)
    probe_repo = f"{ADAPTER_REPO}-probe"
    print(f"[full] no local probe adapter; seeding from HF: {probe_repo}")
    files = HfApi(token=HF_TOKEN).list_repo_files(repo_id=probe_repo)
    # probe pushes 'checkpoint-final' (probe step count < save_steps so no numeric
    # checkpoint fires). Prefer the highest numeric checkpoint if any exist, else final.
    numeric = []
    for f in files:
        if f.startswith("checkpoint-"):
            tag = f.split("/", 1)[0].split("-", 1)[1]
            if tag.isdigit():
                numeric.append(int(tag))
    folder = f"checkpoint-{max(numeric)}" if numeric else "checkpoint-final"
    if folder == "checkpoint-final" and not any(f.startswith("checkpoint-final/") for f in files):
        raise RuntimeError(f"No probe adapter (numeric or final) at {probe_repo} and no local copy")
    probe_local = Path(args.ckpt_root) / "sft_seed_from_probe"
    probe_local.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=probe_repo, allow_patterns=[f"{folder}/*"],
                      local_dir=str(probe_local), token=HF_TOKEN)
    return str(probe_local / folder)


# ============================================================
# 5. Resume / seed logic
# ============================================================
# Full-phase HF checkpoints (this phase's own mid-train saves) take priority:
# they survive a 12h-cap pod death AND already encode the probe-seeded weights
# (the full run started from the probe adapter, so its evolved LoRA delta is
# self-contained). Only on a genuinely fresh full start do we seed from probe.
resume_path = _try_resume_from_hf(ADAPTER_CKPT_REPO, HF_TOKEN, CKPT_DIR) if args.phase == "full" else None

if args.phase == "full" and resume_path is None:
    # Fresh full start: bake probe adapter into a trainable PeftModel, then let
    # SFTTrainer extend it (peft_config=None below since model is already PEFT).
    try:
        seed_ckpt = _find_probe_seed()
        model = PeftModel.from_pretrained(model, seed_ckpt, is_trainable=True)
        print(f"[full] probe adapter attached from {seed_ckpt}; extending training")
    except Exception as e:
        raise SystemExit(f"FATAL: phase=full needs a completed probe but seeding failed: {e}")

# peft_config given to the trainer ONLY when the model isn't already a PeftModel:
#   probe                -> wrap fresh LoRA
#   full + resume        -> wrap fresh LoRA; resume_from_checkpoint loads weights
#   full + fresh (seeded)-> model already PEFT, pass None
trainer_peft_config = None if (args.phase == "full" and resume_path is None) else lora_cfg


# ============================================================
# 6. Train
# ============================================================
trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=ds,
    processing_class=tokenizer,
    formatting_func=format_for_chat,
    data_collator=collator,
    peft_config=trainer_peft_config,
    callbacks=callbacks,
)

print("=" * 60)
print(f"SFT phase={args.phase}  {'(RESUMED)' if resume_path else '(FRESH)'}")
print(f"  base model:  {BASE_MODEL}")
print(f"  dataset:     {DATA_FILE} ({len(ds)} ex)")
print(f"  max_steps:   {max_steps}   epochs={num_train_epochs}")
print(f"  lr:          {TRAIN['learning_rate']}")
print(f"  lora.r:      {LORA['r']}   alpha={LORA['alpha']}")
print(f"  adapter repo: {ADAPTER_CKPT_REPO}")
print("=" * 60, flush=True)

trainer.train(resume_from_checkpoint=resume_path) if resume_path else trainer.train()


# ============================================================
# 7. Save final adapter
# ============================================================
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
trainer.model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print(f"[save] adapter -> {ADAPTER_DIR}")

# For probe: also push the final adapter to the -probe HF repo so a full-phase
# relaunch in a fresh pod (where /tmp was wiped) can still seed from it.
if args.phase == "probe" and HF_TOKEN:
    try:
        api = HfApi(token=HF_TOKEN)
        probe_repo = f"{ADAPTER_REPO}-probe"
        api.create_repo(probe_repo, private=True, exist_ok=True)
        # path_in_repo=checkpoint-final so _find_probe_seed's list works too
        api.upload_folder(folder_path=str(ADAPTER_DIR),
                          path_in_repo="checkpoint-final", repo_id=probe_repo)
        print(f"[save] probe adapter pushed -> {probe_repo}/checkpoint-final")
    except Exception as e:
        print(f"WARN: probe adapter HF push failed (local copy still usable same-pod): {e}")


# ============================================================
# 8. Merge + push (full phase only — Kaggle needs the merged model)
# ============================================================
if args.phase == "full" and HF_TOKEN:
    print("[merge] freeing training memory ...")
    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[merge] loading base {BASE_MODEL} in fp16 ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    merged = peft_model.merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"[merge] merged model -> {MERGED_DIR}")

    print(f"[push] uploading to https://huggingface.co/{MERGED_REPO}")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(MERGED_REPO, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(MERGED_DIR),
        repo_id=MERGED_REPO,
        repo_type="model",
        token=HF_TOKEN,
    )
    print(f"[done] Kaggle inference: set EXPERIMENT config.model_id = '{MERGED_REPO}'")
else:
    print(f"[skip] merge+push (phase={args.phase}, has_token={bool(HF_TOKEN)})")

print("ALL DONE.")
