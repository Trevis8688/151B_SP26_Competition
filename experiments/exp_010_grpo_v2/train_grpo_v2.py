"""GRPO v2 training script — DSMLP batch-mode friendly.

Ported from exp_009/train_grpo.ipynb. Changes from exp_009:
  - LR 2e-5 → 1e-5 (config.json)
  - BETA 0.01 → 0.02 (config.json)
  - curriculum: sweet_spot_ids_clean.json (strict-70) — same as exp_009 best
  - epochs: 2
  - save: Drive callback removed; HF push-on-save every 10 steps to ckpt repo
  - HF push at end (replaces Cell 10/11 of exp_009 notebook)

Run 2 v3 (torch 2.6 + vLLM 0.8 for Qwen3 support):
  - torch 2.5 → 2.6 (vllm 0.6.6 doesn't support Qwen3ForCausalLM)
  - vllm 0.6.6 → 0.8.5 (first stable with Qwen3 support)
  - flash-attn rebuilt against torch 2.6 (--no-build-isolation needed)
  - 4-bit BnB removed; base loads in bf16/fp16 (vLLM can't read BnB).
  - use_vllm=True, vllm_mode="colocate" → ~3-5x faster rollouts vs HF generate.
  - Gradient checkpointing kept ON (tight memory on a5000 24GB next to vLLM).
  - Flash Attention 2 explicit (attn_implementation="flash_attention_2").

Usage (DSMLP, a5000 single GPU, 12hr container).
Install order is load-bearing — install the three groups separately:
  1. requirements.txt: torch 2.6 + bitsandbytes/trl/peft/accelerate/etc.
  2. flash-attn 2.7.x with --no-build-isolation (sees the freshly-installed torch)
  3. vllm 0.8.5 with --no-deps (don't let it pull its own torch)

    K8S_TIMEOUT_SECONDS=43200 launch.sh -g 1 -v a5000 -m 48 -c 8 -B \\
        -- bash -c 'cd /home/$USER/151B_SP26_Competition && \\
                    git pull origin main && \\
                    pip install -q -r experiments/exp_010_grpo_v2/requirements.txt && \\
                    pip install -q --no-build-isolation flash-attn==2.7.4.post1 && \\
                    pip install -q --no-deps vllm==0.8.5 && \\
                    HF_TOKEN=$(cat /home/$USER/.hf_token) \\
                        python experiments/exp_010_grpo_v2/train_grpo_v2.py'

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

# Mock the *optional* trl deps that aren't installed (mergekit, llm_blender,
# vllm_ascend). vllm itself is now a real install (use_vllm=True), so it must
# NOT be mocked. ModuleSpec is needed for transformers' importlib.util.find_spec
# on Python 3.11 (bare MagicMock fails with: __spec__ is not set).
def _mock(name):
    m = MagicMock()
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = m

for mod in [
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
    AutoModelForCausalLM, AutoTokenizer, TrainerCallback
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import HfApi

# Surface available vLLM-related GRPOConfig fields once so we can spot API
# drift between TRL versions in the log (helps debug if a vllm_* kwarg fails).
import dataclasses
_vllm_fields = [f.name for f in dataclasses.fields(GRPOConfig) if "vllm" in f.name.lower()]
print(f"GRPOConfig vLLM fields: {_vllm_fields}")

# ============================================================
# vLLM EngineArgs compat shim
# TRL 0.21 passes kwargs (model_impl, possibly others) that vllm 0.6.6.post1's
# EngineArgs doesn't accept. vllm 0.6.6 is pinned because it's the last release
# supporting torch==2.5.1. Filter unknown kwargs so the LLM() construction in
# GRPOTrainer succeeds. For model_impl specifically: vllm 0.6 always uses its
# native engine (== model_impl="vllm" in 0.8+), so dropping the kwarg preserves
# behavior. If we ever swap in a newer vllm/torch, this whole block becomes a
# no-op (the kwargs all match).
# ============================================================
try:
    import inspect
    import vllm.engine.arg_utils as _au
    _orig_engine_init = _au.EngineArgs.__init__
    _allowed_engine_kwargs = set(inspect.signature(_orig_engine_init).parameters)
    def _patched_engine_init(self, *args, **kwargs):
        dropped = {k: kwargs.pop(k) for k in list(kwargs) if k not in _allowed_engine_kwargs}
        if dropped:
            print(f"[vllm-compat] dropped unsupported EngineArgs kwargs: {list(dropped)}",
                  flush=True)
        return _orig_engine_init(self, *args, **kwargs)
    _au.EngineArgs.__init__ = _patched_engine_init
    print(f"[vllm-compat] EngineArgs has {len(_allowed_engine_kwargs)} known fields; "
          f"unknown kwargs will be filtered.")
except Exception as e:
    print(f"[vllm-compat] could NOT patch EngineArgs (use_vllm may fail): {e}")

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
# 3. Tokenizer + fp16/bf16 base + LoRA wrap
# vLLM colocate mode reads weights directly from the trainer's PEFT model, so
# the base must be in a vLLM-compatible format (bf16/fp16, not BnB 4-bit).
# Memory is tight on a5000 24GB: ~8GB base + ~11GB vLLM (gpu_mem_util=0.45) +
# ~3-5GB activations/optim. Gradient checkpointing stays ON.
# ============================================================
print(f"\nLoading tokenizer + {COMPUTE_DTYPE} model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# attn_implementation="flash_attention_2" requested explicitly; transformers
# falls back to SDPA if the flash-attn wheel didn't build (we'll see a warning
# in the log if so — at that point we should suspect the install step).
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=COMPUTE_DTYPE,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
# Replaces prepare_model_for_kbit_training (which was BnB-specific).
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

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

# Build GRPOConfig kwargs. vllm_* fields are added conditionally — TRL renamed
# some between 0.16 and 0.21 (vllm_gpu_memory_utilization, vllm_mode, etc).
# We only pass fields that the installed TRL actually defines, so the script
# survives minor version drift.
grpo_kwargs = dict(
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
    use_vllm=TRAIN.get("use_vllm", False),
)
_vllm_field_names = set(_vllm_fields)
if TRAIN.get("use_vllm"):
    for k, v in (
        ("vllm_mode", TRAIN.get("vllm_mode", "colocate")),
        ("vllm_gpu_memory_utilization", TRAIN.get("vllm_gpu_memory_utilization", 0.45)),
        ("vllm_tensor_parallel_size", 1),
        ("vllm_dtype", "auto"),
    ):
        if k in _vllm_field_names:
            grpo_kwargs[k] = v
        else:
            print(f"  (skipping unsupported GRPOConfig field: {k}={v})")
training_args = GRPOConfig(**grpo_kwargs)

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
    """Push the LoRA adapter to HF Hub after each save_steps save. Survives
    DeadlineExceeded — without this, if the pod hits 12hr timeout mid-merge,
    everything is lost. With it, we can recover from any saved checkpoint.

    Pushes only adapter files (~130 MB) to a dedicated checkpoints repo.
    Each checkpoint goes under its own subfolder so they don't overwrite.
    """
    def __init__(self, hf_token, ckpt_repo):
        self.api = HfApi(token=hf_token)
        self.repo = ckpt_repo
        try:
            self.api.create_repo(ckpt_repo, private=True, exist_ok=True)
            print(f"HF adapter-checkpoint repo ready: {ckpt_repo}", flush=True)
        except Exception as e:
            print(f"⚠ create_repo({ckpt_repo}) failed: {e}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.exists():
            return
        try:
            self.api.upload_folder(
                folder_path=str(ckpt_dir),
                path_in_repo=f"checkpoint-{state.global_step}",
                repo_id=self.repo,
                allow_patterns=["adapter_*.json", "adapter_*.safetensors",
                                "README.md", "training_args.bin"],
            )
            print(f"  ↑ pushed checkpoint-{state.global_step} → "
                  f"https://huggingface.co/{self.repo}/tree/main/checkpoint-{state.global_step}",
                  flush=True)
        except Exception as e:
            # Never let a push failure kill training.
            print(f"  ⚠ HF push failed for checkpoint-{state.global_step}: {e}",
                  flush=True)

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
