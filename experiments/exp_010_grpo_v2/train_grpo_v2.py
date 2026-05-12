"""GRPO v2 training script — DSMLP batch-mode friendly.

Ported from exp_009/train_grpo.ipynb. Changes from exp_009:
  - LR 2e-5 → 1e-5 (config.json)
  - BETA 0.01 → 0.02 (config.json)
  - curriculum: sweet_spot_ids_clean.json (strict-70) — same as exp_009 best
  - epochs: 2
  - save: Drive callback removed; HF push-on-save every 10 steps to ckpt repo
  - HF push at end (replaces Cell 10/11 of exp_009 notebook)

Run 2 v4 (HF native + split-container resume):
  - vLLM path abandoned (vllm 0.6 doesn't support Qwen3; vllm 0.8 has cascading
    optional-dep import failures even with --no-deps).
  - Back to torch 2.5.1 + 4-bit BnB + HF model.generate (exp_009 known-good stack).
  - epochs=2 won't fit one 12h container; HFPushAdapterCallback now saves the
    FULL trainer state (adapter + optimizer + scheduler + RNG + trainer_state)
    so a second container can resume_from_checkpoint and finish epoch 2.
  - Container 1 (tonight): runs to step ~70 or DeadlineExceeded, whichever first.
  - Container 2 (next morning): script auto-detects latest HF checkpoint, pulls
    it down, resumes mid-training. Total elapsed: ~20-24h across two containers.

Usage (DSMLP, a5000 single GPU, 12hr container):
    K8S_TIMEOUT_SECONDS=43200 launch.sh -g 1 -v a5000 -m 48 -c 8 -B \\
        -- bash -c 'cd /home/$USER/151B_SP26_Competition && \\
                    git pull origin main && \\
                    pip install -q -r experiments/exp_010_grpo_v2/requirements.txt && \\
                    HF_TOKEN=$(cat /home/$USER/.hf_token) \\
                        python experiments/exp_010_grpo_v2/train_grpo_v2.py'

To force fresh start (ignore HF checkpoints), set DISABLE_RESUME=1.

Monitor:
    kubectl logs -f <pod_name>

Requires env var HF_TOKEN.
Expects repo layout:
    <repo>/data/public.jsonl
    <repo>/data/splits/dev.jsonl
    <repo>/judger.py
    <repo>/utils.py
    <repo>/experiments/exp_010_grpo_v2/{config.json, prompts.py, sweet_spot_ids_clean.json}
"""
import json
import os
import sys
import shutil
import gc
import re
import random
import importlib.machinery
from pathlib import Path
from unittest.mock import MagicMock

# ============================================================
# 0. Environment — set BEFORE importing torch / trl
# ============================================================
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# (BnB env var removed — Run 2 v3 doesn't use 4-bit BnB, weights are bf16 for vLLM)

# Run 2 v4: use_vllm=False, so we mock vllm + its tree of optional imports
# so TRL doesn't blow up trying to import them. ModuleSpec is needed for
# transformers' importlib.util.find_spec on Python 3.11.
def _mock(name):
    m = MagicMock()
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = m

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
    _mock(mod)

# ============================================================
# 1. Paths + config
# ============================================================
EXP_DIR  = Path(__file__).resolve().parent
REPO_DIR = EXP_DIR.parent.parent

CONFIG = json.loads((EXP_DIR / "config.json").read_text())
CURRICULUM_FILE = CONFIG.get("curriculum_file", "sweet_spot_ids.json")
CURRICULUM = json.loads((EXP_DIR / CURRICULUM_FILE).read_text())
SWEET_IDS  = set(CURRICULUM["sweet_ids"])

PUBLIC_JSONL = REPO_DIR / "data" / "public.jsonl"
DEV_JSONL    = REPO_DIR / "data" / "splits" / "dev.jsonl"

# Repo root on sys.path so we can import judger + utils + prompts
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(EXP_DIR))

print(f"REPO_DIR:       {REPO_DIR}")
print(f"EXP_DIR:        {EXP_DIR}")
print(f"Curriculum:     {len(SWEET_IDS)} prompts from {CURRICULUM_FILE}")
print(f"Output repo:    {CONFIG['output_repo']}")
print(f"Adapter ckpt repo: {CONFIG.get('adapter_checkpoints_repo', '(none)')}")

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
from huggingface_hub import HfApi, snapshot_download

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
# Run 2 v4: back to BnB 4-bit (matches exp_009 known-good setup). On a5000 24GB
# this gives plenty of headroom for HF generate rollouts at max_completion=4096.
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
# (Removed CKPT_DIR.rmtree — we may want to keep checkpoints across re-runs.
#  The resume path writes its downloads into CKPT_DIR before trainer.train.)

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

class HFPushAdapterCallback(TrainerCallback):
    """Push the FULL trainer checkpoint to HF Hub after each save_steps save.
    Run 2 v4 needs split-container resume: container 1 hits DeadlineExceeded
    around step 70, container 2 next morning pulls the latest checkpoint and
    calls trainer.train(resume_from_checkpoint=...) to finish epoch 2.

    For HF Trainer's resume to work correctly we need: adapter weights + config
    + optimizer state + LR scheduler state + RNG state + trainer_state.json.
    Total per-checkpoint upload is ~250MB (vs ~130MB adapter-only).
    """
    def __init__(self, hf_token, ckpt_repo):
        self.api = HfApi(token=hf_token)
        self.repo = ckpt_repo
        try:
            self.api.create_repo(ckpt_repo, private=True, exist_ok=True)
            print(f"HF checkpoint repo ready: {ckpt_repo}", flush=True)
        except Exception as e:
            print(f"⚠ create_repo({ckpt_repo}) failed: {e}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.exists():
            return
        try:
            # ignore_patterns instead of allow_patterns: push everything except
            # bulky files we don't need. Specifically excludes any merged-model
            # safetensors that might accidentally end up in the ckpt dir.
            self.api.upload_folder(
                folder_path=str(ckpt_dir),
                path_in_repo=f"checkpoint-{state.global_step}",
                repo_id=self.repo,
                ignore_patterns=["*.bin.tmp", "global_step*"],
            )
            print(f"  ↑ pushed checkpoint-{state.global_step} (full state) → "
                  f"https://huggingface.co/{self.repo}/tree/main/checkpoint-{state.global_step}",
                  flush=True)
        except Exception as e:
            # Never let a push failure kill training.
            print(f"  ⚠ HF push failed for checkpoint-{state.global_step}: {e}",
                  flush=True)


def _try_resume_from_hf(ckpt_repo, hf_token, local_ckpt_dir):
    """Look for the latest checkpoint-N folder in ckpt_repo. If found and
    DISABLE_RESUME is not set, download to local_ckpt_dir and return the
    local path for trainer.train(resume_from_checkpoint=...). Otherwise None.
    """
    if os.environ.get("DISABLE_RESUME"):
        print("DISABLE_RESUME set — skipping HF checkpoint lookup.")
        return None
    if not ckpt_repo or not hf_token:
        print("No adapter_checkpoints_repo or HF_TOKEN — fresh start.")
        return None
    api = HfApi(token=hf_token)
    try:
        # repo_exists raises if repo doesn't exist; create_repo with exist_ok
        # also won't error if it exists, so we try to list its tree first
        files = api.list_repo_files(repo_id=ckpt_repo)
    except Exception as e:
        print(f"No prior HF checkpoints found at {ckpt_repo} (or repo new): {e}")
        return None
    # Find unique checkpoint-N subdirs
    steps = set()
    for f in files:
        if f.startswith("checkpoint-"):
            try:
                steps.add(int(f.split("/", 1)[0].split("-", 1)[1]))
            except (ValueError, IndexError):
                pass
    if not steps:
        print(f"HF repo {ckpt_repo} exists but has no checkpoint-N folders — fresh start.")
        return None
    latest = max(steps)
    target_path = Path(local_ckpt_dir) / f"checkpoint-{latest}"
    print(f"\n🔄 RESUMING from HF checkpoint-{latest} (downloading to {target_path}) ...")
    snapshot_download(
        repo_id=ckpt_repo,
        allow_patterns=[f"checkpoint-{latest}/*"],
        local_dir=str(local_ckpt_dir),
        token=hf_token,
    )
    print(f"  ↓ checkpoint-{latest} downloaded. resume_from_checkpoint={target_path}")
    return str(target_path)

if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

callbacks = [RewardLogCallback()]
ADAPTER_CKPT_REPO = CONFIG.get("adapter_checkpoints_repo")
if ADAPTER_CKPT_REPO and HF_TOKEN:
    callbacks.append(HFPushAdapterCallback(HF_TOKEN, ADAPTER_CKPT_REPO))
else:
    print("Note: HF push-on-save disabled (no adapter_checkpoints_repo or HF_TOKEN).",
          flush=True)

# ============================================================
# 7. Train
# ============================================================
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward, format_reward],
    args=training_args,
    train_dataset=train_dataset,
    callbacks=callbacks,
)

steps_per_epoch = len(train_dataset) // (TRAIN["per_device_train_batch_size"]
                                         * TRAIN["gradient_accumulation_steps"])
total_steps = steps_per_epoch * TRAIN["epochs"]
tokens_per_epoch = (len(train_dataset) * TRAIN["num_generations"]
                    * TRAIN["max_completion_length"]) / 1e6

# Resume lookup: if a prior checkpoint lives at ADAPTER_CKPT_REPO on HF,
# download it and pass to trainer.train(resume_from_checkpoint=...). On a
# fresh run (no prior checkpoints) this returns None and we start clean.
resume_path = _try_resume_from_hf(ADAPTER_CKPT_REPO, HF_TOKEN, CKPT_DIR)

print("\n" + "=" * 60)
print("Starting GRPO training" + ("  (RESUMED)" if resume_path else "  (FRESH)"))
print(f"  Epochs:           {TRAIN['epochs']}")
print(f"  Steps/epoch:      ~{steps_per_epoch}")
print(f"  Total steps:      ~{total_steps}")
print(f"  Tokens/epoch:     ~{tokens_per_epoch:.1f}M")
print(f"  LR:               {TRAIN['learning_rate']}")
print(f"  BETA:             {TRAIN['beta']}")
print(f"  num_generations:  {TRAIN['num_generations']}")
print(f"  max_completion:   {TRAIN['max_completion_length']}")
if resume_path:
    print(f"  Resuming from:    {resume_path}")
print("=" * 60 + "\n", flush=True)

trainer.train(resume_from_checkpoint=resume_path) if resume_path else trainer.train()

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
