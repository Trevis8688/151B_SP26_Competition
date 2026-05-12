"""Sanity-check pilot: 5 GRPO steps on 10 prompts, no HF push.

Validates the Run 2 v2 stack (vLLM + fp16 base + FA2) without burning the 12hr
container. If this finishes without OOM and reward fluctuates, the real run
should work.

Usage (DSMLP, inside the container):
    pip install -q -r experiments/exp_010_grpo_v2/requirements.txt
    pip install -q --no-deps vllm==0.6.6.post1
    python experiments/exp_010_grpo_v2/pilot.py

What to look for in the log:
  1. "GRPOConfig vLLM fields:" — confirms TRL exposes vllm_mode etc.
  2. "Loading {model} in torch.bfloat16" — confirms fp16/bf16 base, no BnB.
  3. vLLM init banner — no OOM at gpu_memory_utilization=0.45.
  4. First [step 1] reward log appears in <5 min.
  5. Per-step wallclock stays under 4 min on steps 2-5 (LoRA sync overhead).

Reads same config.json as train_grpo_v2.py but overrides epochs/steps/max_completion.
"""
import os, sys, json, gc, random
import importlib.machinery
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("BNB_CUDA_VERSION", "124")

# Bare MagicMock fails newer transformers' importlib.util.find_spec() check
# because __spec__ auto-mocks to a MagicMock (no _initializing attr) → ValueError.
# Give each mock a real ModuleSpec so find_spec returns a valid value.
def _mock(name):
    m = MagicMock()
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = m

# vllm itself is now a real install (use_vllm=True) — do NOT mock.
for mod in ["vllm_ascend", "vllm_ascend.distributed",
            "vllm_ascend.distributed.device_communicators",
            "vllm_ascend.distributed.device_communicators.pyhccl",
            "mergekit", "mergekit.config", "mergekit.merge", "llm_blender"]:
    _mock(mod)

EXP_DIR  = Path(__file__).resolve().parent
REPO_DIR = EXP_DIR.parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(EXP_DIR))

CONFIG     = json.loads((EXP_DIR / "config.json").read_text())
CURRICULUM = json.loads((EXP_DIR / CONFIG.get("curriculum_file", "sweet_spot_ids_clean.json")).read_text())
SWEET_IDS  = set(CURRICULUM["sweet_ids"])

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"CUDA: torch={torch.__version__}  cuda={torch.version.cuda}  bf16={torch.cuda.is_bf16_supported()}")

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import dataclasses
_vllm_fields = [f.name for f in dataclasses.fields(GRPOConfig) if "vllm" in f.name.lower()]
print(f"GRPOConfig vLLM fields: {_vllm_fields}")
from judger import Judger
from prompts import SYSTEM_PROMPT_MATH, SYSTEM_PROMPT_MCQ, FEWSHOT_MATH, FEWSHOT_MCQ

BF16_OK = torch.cuda.is_bf16_supported()
CDT = torch.bfloat16 if BF16_OK else torch.float16

print(f"\nLoading {CONFIG['base_model']} in {CDT} (no BnB; vLLM-compatible) ...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"], torch_dtype=CDT, device_map="auto",
    trust_remote_code=True, attn_implementation="flash_attention_2",
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
T = CONFIG["training"]
model = get_peft_model(model, LoraConfig(
    r=T["lora_r"], lora_alpha=T["lora_alpha"], lora_dropout=0.0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"))
model.print_trainable_parameters()

# --- 10-prompt mini dataset ---
import re
JUDGER = Judger()
LETTERS = "ABCDEFGHIJ"
rows = []
with open(REPO_DIR / "data" / "public.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        if ex["id"] not in SWEET_IDS: continue
        is_mcq = bool(ex.get("options"))
        qt = ex["question"]
        if is_mcq:
            qt += "\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {v}" for i, v in enumerate(ex["options"]))
            msgs = [{"role":"system","content":SYSTEM_PROMPT_MCQ}] + FEWSHOT_MCQ + [{"role":"user","content":qt}]
        else:
            msgs = [{"role":"system","content":SYSTEM_PROMPT_MATH}] + FEWSHOT_MATH + [{"role":"user","content":qt}]
        rows.append({"prompt": msgs, "answer_json": json.dumps(ex["answer"]),
                     "options_json": json.dumps(ex.get("options", [])),
                     "is_mcq": is_mcq, "id": ex["id"]})
        if len(rows) == 10: break
print(f"Pilot dataset: {len(rows)} prompts")
ds = Dataset.from_list(rows)

# --- rewards ---
_BOXED_RE = re.compile(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
def post(t):
    i = t.rfind("</think>"); return t if i == -1 else t[i+8:]
def correctness(prompts, completions, answer_json=None, options_json=None, is_mcq=None, **kw):
    rs = []
    for i, c in enumerate(completions):
        t = "".join(m.get("content","") for m in c) if isinstance(c, list) else str(c)
        try:
            gold = json.loads(answer_json[i]); opts = json.loads(options_json[i])
            ok = JUDGER.auto_judge(post(t), gold, ([opts]*len(gold)) if gold else [None])
        except Exception:
            ok = False
        rs.append(1.0 if ok else 0.0)
    return rs
def fmt(prompts, completions, **kw):
    rs = []
    for c in completions:
        t = "".join(m.get("content","") for m in c) if isinstance(c, list) else str(c)
        r = 0.0
        if "</think>" in t: r += 0.05
        b_all = _BOXED_RE.findall(t)
        if b_all: r += 0.10
        b_post = _BOXED_RE.findall(post(t))
        if "</think>" in t and b_post: r += 0.05
        if len(b_post) == 1: r += 0.025
        rs.append(r)
    return rs

# --- monkey-patch generate ---
if hasattr(type(model), "generate"):
    model.generate = type(model).generate.__get__(model)
_og = model.generate
def _eg(*a, **k):
    model.eval(); o = _og(*a, **k); model.train(); return o
model.generate = _eg

class Logger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            r = {k:v for k,v in logs.items() if any(s in k for s in ("loss","reward","kl"))}
            if r:
                print(f"  [step {state.global_step}] " + "  ".join(
                    f"{k}={v:.4f}" if isinstance(v,float) else f"{k}={v}" for k,v in r.items()), flush=True)

if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

# OVERRIDE: max 5 steps regardless of dataset size
pilot_kwargs = dict(
    output_dir=str(EXP_DIR / "pilot_checkpoints"),
    max_steps=5, num_train_epochs=1,
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    learning_rate=T["learning_rate"], beta=T["beta"], optim="adamw_8bit",
    bf16=BF16_OK, fp16=not BF16_OK, logging_steps=1,
    save_strategy="no", report_to="none", seed=42,
    num_generations=T["num_generations"],
    max_prompt_length=T["max_prompt_length"],
    max_completion_length=2048,  # short for pilot
    temperature=T["temperature_train"],
    use_vllm=T.get("use_vllm", False),
)
_vff = set(_vllm_fields)
if T.get("use_vllm"):
    for k, v in (
        ("vllm_mode", T.get("vllm_mode", "colocate")),
        ("vllm_gpu_memory_utilization", T.get("vllm_gpu_memory_utilization", 0.45)),
        ("vllm_tensor_parallel_size", 1),
        ("vllm_dtype", "auto"),
    ):
        if k in _vff:
            pilot_kwargs[k] = v
        else:
            print(f"  (skipping unsupported field: {k}={v})")
args = GRPOConfig(**pilot_kwargs)
trainer = GRPOTrainer(model=model, processing_class=tokenizer,
                     reward_funcs=[correctness, fmt], args=args,
                     train_dataset=ds, callbacks=[Logger()])
print("\n" + "="*60)
print(f"Pilot: 5 steps, {T['num_generations']} gens, max_completion=2048")
print("If this finishes without OOM and you see reward fluctuating, the stack is good.")
print("="*60 + "\n", flush=True)
trainer.train()
print("\n✅ Pilot complete. Stack works.  Now run train_grpo_v2.py for the real thing.")
