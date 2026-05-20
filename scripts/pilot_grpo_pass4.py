"""GRPO pass-4 PILOT — measure dead-step rate + memory across (G, T, max_completion).

WHY THIS EXISTS
---------------
Pass-2 and pass-3 GRPO each had only ~10% of steps produce a useful gradient
(reward_std>0). We confirmed (TRL 0.21 source) that top_k was already disabled
(GRPOConfig top_k=None, top_p=1.0) in those runs — so the low within-group
variance is NOT a sampler-truncation artifact; it is the 4-bit-quantized pass-N
policy being peaked/confident. The two quality-neutral / low-risk levers left are:
  - num_generations bump (G): binomial-guaranteed more variance, no quality risk.
  - temperature bump (T): flattens the peaked policy; must verify it raises
    entropy / lowers reward_std=0 WITHOUT wrecking rollout coherence.

This pilot runs a handful of GRPO steps under each candidate recipe and reports,
per config: mean entropy, mean reward_std, frac_reward_zero_std (the dead-step
rate — the headline number), mean completion length, clip rate, KL, and whether
it OOM'd. That table picks pass-4's recipe. WITHOUT these numbers, choosing T is
a guess.

MEMORY FAITHFULNESS
-------------------
Real pass-N training uses per_device_train_batch_size=1 (the entropy_from_logits
OOM that killed exp_010 scales with pdbs*seqlen*vocab, NOT with G). So the pilot
keeps pdbs=1 and sets gradient_accumulation_steps to a multiple of G — TRL
requires generation_batch_size (= pdbs*world*grad_accum) % num_generations == 0.
G=6 therefore uses grad_accum=6; G=4 uses grad_accum=4. Each optimizer step then
processes exactly one prompt's group of G generations (gen_batch/G = 1).

CONFIGS (override count per config with PILOT_STEPS, default 6):
  1. G=4, T=1.0, max_completion=4096   (reproduces pass-3 — reference / harness check)
  2. G=6, T=1.0, max_completion=3584   (the primary lever: group-size bump)
  3. G=6, T=1.1, max_completion=3584   (adds the temperature lever)

Prompt pool: exp_019's curriculum_pass3.json (88 strict prompts). NOTE this was
sampled from the PASS-2 model, so absolute reward_std is approximate — but the
RELATIVE comparison across configs (which is all the pilot needs) is valid on any
reasonable prompt set. The real pass-4 curriculum (pass-3-sampled) comes later.

Runtime: ~10 min/optimizer step at these lengths (HF generate, no continuous
batching) → ~60 min/config at PILOT_STEPS=6, ~3h total for 3 configs. Lower
PILOT_STEPS to 3-4 for a faster (noisier) read. Batch-mode friendly; no HF push.

Usage (DSMLP, via scripts/launch_pilot_pass4.sh):
    HF_TOKEN=$(cat ~/.hf_token) python scripts/pilot_grpo_pass4.py
    PILOT_STEPS=4 HF_TOKEN=$(cat ~/.hf_token) python scripts/pilot_grpo_pass4.py
    # add a 4th config:  EXTRA_CONFIG="8,2560,1.0"  (G,max_completion,T)
"""
import json, os, sys, gc, re, random, importlib.machinery
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# use_vllm=False → mock vllm + its optional-import tree before importing trl.
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
    "mergekit", "mergekit.config", "mergekit.merge", "llm_blender",
]:
    _mock(mod)

# ---- paths ----
REPO_DIR = Path(__file__).resolve().parent.parent
EXP019   = REPO_DIR / "experiments" / "exp_019_grpo_pass3"
CURRICULUM_PATH = EXP019 / "curriculum_pass3.json"
PUBLIC_JSONL = REPO_DIR / "data" / "public.jsonl"
DEV_JSONL    = REPO_DIR / "data" / "splits" / "dev.jsonl"
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(EXP019))

BASE_MODEL  = os.environ.get("PILOT_BASE_MODEL", "TrevorDuong/qwen3-4b-thinking-grpo-pass3")
PILOT_STEPS = int(os.environ.get("PILOT_STEPS", "6"))

# Candidate recipes: (num_generations, max_completion_length, temperature).
CONFIGS = [
    (4, 4096, 1.0),
    (6, 3584, 1.0),
    (6, 3584, 1.1),
]
if os.environ.get("EXTRA_CONFIG"):
    g, mc, t = os.environ["EXTRA_CONFIG"].split(",")
    CONFIGS.append((int(g), int(mc), float(t)))

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from judger import Judger  # noqa: E402
from prompts import SYSTEM_PROMPT_MATH, SYSTEM_PROMPT_MCQ, FEWSHOT_MATH, FEWSHOT_MCQ  # noqa: E402

BF16_OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if BF16_OK else torch.float16
RANDOM_SEED = 42
random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"bf16 supported: {BF16_OK} → compute dtype {COMPUTE_DTYPE}")
print(f"Base model: {BASE_MODEL}")
print(f"Pilot steps per config: {PILOT_STEPS}")
print(f"Configs (G, max_completion, T): {CONFIGS}\n")

# ---- prompt pool ----
SWEET_IDS = set(json.loads(CURRICULUM_PATH.read_text())["sweet_ids"])
dev_ids = {json.loads(l)["id"] for l in open(DEV_JSONL)}
LETTERS = "ABCDEFGHIJ"
JUDGER = Judger()

rows = []
with open(PUBLIC_JSONL) as f:
    for line in f:
        ex = json.loads(line)
        if ex["id"] in dev_ids or ex["id"] not in SWEET_IDS:
            continue
        is_mcq = bool(ex.get("options"))
        qt = ex["question"]
        if is_mcq:
            qt += "\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {v}" for i, v in enumerate(ex["options"]))
            sys_prompt, fewshots = SYSTEM_PROMPT_MCQ, FEWSHOT_MCQ
        else:
            sys_prompt, fewshots = SYSTEM_PROMPT_MATH, FEWSHOT_MATH
        msgs = [{"role": "system", "content": sys_prompt}] + fewshots + [{"role": "user", "content": qt}]
        rows.append({
            "prompt": msgs,
            "answer_json": json.dumps(ex["answer"]),
            "options_json": json.dumps(ex.get("options", [])),
            "is_mcq": is_mcq, "id": ex["id"],
        })
random.shuffle(rows)
train_dataset = Dataset.from_list(rows)
print(f"Prompt pool: {len(rows)} prompts ({sum(r['is_mcq'] for r in rows)} MCQ, "
      f"{sum(not r['is_mcq'] for r in rows)} FF)\n")

# ---- reward funcs (identical to exp_019 train_grpo.py) ----
_BOXED_RE = re.compile(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")

def extract_post_think(text):
    idx = text.rfind("</think>")
    return text if idx == -1 else text[idx + len("</think>"):]

def correctness_reward(prompts, completions, answer_json=None, options_json=None, is_mcq=None, **kwargs):
    out = []
    for i, comp in enumerate(completions):
        text = "".join(m.get("content", "") for m in comp) if isinstance(comp, list) else str(comp)
        post = extract_post_think(text)
        gold = json.loads(answer_json[i]) if answer_json else []
        opts = json.loads(options_json[i]) if options_json else []
        try:
            ok = JUDGER.auto_judge(post, gold, ([opts] * len(gold)) if gold else [None])
        except Exception:
            ok = False
        out.append(1.0 if ok else 0.0)
    return out

_LB_MAX, _LB_CAP = 0.05, 16384
def _length_bonus(text):
    n = len(text)
    return 0.0 if n >= _LB_CAP else _LB_MAX * (1.0 - n / _LB_CAP)

def format_reward(prompts, completions, **kwargs):
    out = []
    for comp in completions:
        text = "".join(m.get("content", "") for m in comp) if isinstance(comp, list) else str(comp)
        r = 0.0
        has_close = "</think>" in text
        if has_close: r += 0.05
        if _BOXED_RE.findall(text): r += 0.10
        post_boxed = _BOXED_RE.findall(extract_post_think(text))
        if has_close and post_boxed: r += 0.05
        if len(post_boxed) == 1: r += 0.025
        r += _length_bonus(text)
        out.append(r)
    return out

# ---- metric collector ----
class MetricCollector(TrainerCallback):
    # Keys we care about; TRL key names vary slightly by version so we substring-match.
    WANT = ("reward", "std", "entropy", "completion", "kl", "clip", "zero")
    def __init__(self):
        self.logs = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        rec = {k: v for k, v in logs.items()
               if any(s in k.lower() for s in self.WANT) and isinstance(v, (int, float))}
        if rec:
            rec["_step"] = state.global_step
            self.logs.append(rec)
            parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()]
            print(f"    [step {state.global_step}] " + "  ".join(parts), flush=True)

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def summarize(collector):
    """Average each tracked metric over the collected steps."""
    keys = set()
    for r in collector.logs:
        keys.update(k for k in r if k != "_step")
    return {k: _mean([r.get(k) for r in collector.logs]) for k in sorted(keys)}, len(collector.logs)

# ---- run each config in isolation (fresh 4-bit model + LoRA per config) ----
def build_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=COMPUTE_DTYPE, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=COMPUTE_DTYPE)
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    m = get_peft_model(m, lora)
    # eval-mode rollout wrap (exp_009 Cell 8 fix — required for correct generation
    # under gradient_checkpointing, else use_cache is disabled and rollouts are slow/wrong).
    if hasattr(type(m), "generate"):
        m.generate = type(m).generate.__get__(m)
    _orig = m.generate
    def _eval_generate(*a, **k):
        m.eval(); out = _orig(*a, **k); m.train(); return out
    m.generate = _eval_generate
    if not hasattr(m, "warnings_issued"):
        m.warnings_issued = {}
    return m, tok

results = {}
for (G, MAXC, T) in CONFIGS:
    tag = f"G{G}_T{T}_mc{MAXC}"
    grad_accum = G  # pdbs=1, world=1 → generation_batch_size = grad_accum; need % G == 0
    print("\n" + "=" * 64)
    print(f"CONFIG {tag}: num_generations={G}  temperature={T}  "
          f"max_completion={MAXC}  grad_accum={grad_accum}  steps={PILOT_STEPS}")
    print("=" * 64, flush=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        model, tokenizer = build_model()
        args = GRPOConfig(
            output_dir=f"/tmp/pilot_{tag}",
            max_steps=PILOT_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            learning_rate=2e-5, lr_scheduler_type="constant", warmup_ratio=0.0,
            optim="adamw_8bit", bf16=BF16_OK, fp16=not BF16_OK,
            logging_steps=1, save_strategy="no", report_to="none", seed=RANDOM_SEED,
            num_generations=G, max_prompt_length=1024, max_completion_length=MAXC,
            temperature=T, beta=0.01, use_vllm=False,
        )
        collector = MetricCollector()
        trainer = GRPOTrainer(
            model=model, processing_class=tokenizer,
            reward_funcs=[correctness_reward, format_reward],
            args=args, train_dataset=train_dataset, callbacks=[collector],
        )
        trainer.train()
        summary, n = summarize(collector)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        results[tag] = {"ok": True, "steps_logged": n, "peak_gb": peak_gb, **summary}
        print(f"  → {tag} done. peak_mem={peak_gb:.1f}GB  metrics={summary}", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        results[tag] = {"ok": False, "error": "OOM", "detail": str(e)[:200]}
        print(f"  ✗ {tag} OOM: {str(e)[:200]}", flush=True)
    except Exception as e:
        results[tag] = {"ok": False, "error": type(e).__name__, "detail": str(e)[:300]}
        print(f"  ✗ {tag} FAILED ({type(e).__name__}): {str(e)[:300]}", flush=True)
    finally:
        for v in ("trainer", "model", "tokenizer"):
            if v in globals():
                del globals()[v]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

# ---- final table ----
print("\n\n" + "#" * 64)
print("PILOT SUMMARY — pick pass-4 recipe from this")
print("#" * 64)
print("\nHeadline metric: frac_reward_zero_std (the DEAD-STEP rate — lower is better).")
print("Also watch: entropy (higher = less peaked), reward_std (higher = more signal),")
print("completion length / clip rate (coherence proxy), peak_gb (must fit ~24GB A5000).\n")
for tag, r in results.items():
    if not r.get("ok"):
        print(f"  {tag:18s}  FAILED: {r.get('error')}  {r.get('detail','')}")
        continue
    def g(*names):
        for n in names:
            for k, v in r.items():
                if n in k.lower() and isinstance(v, float):
                    return v
        return None
    zero = g("zero")           # frac_reward_zero_std
    ent  = g("entropy")
    rstd = g("reward_std", "std")
    clen = g("completion", "length")
    clip = g("clip")
    print(f"  {tag:18s}  dead_step_frac={zero!s:>8.8}  entropy={ent!s:>8.8}  "
          f"reward_std={rstd!s:>8.8}  comp_len={clen!s:>8.8}  clip={clip!s:>8.8}  "
          f"peak={r.get('peak_gb')!s:>5.5}GB  (steps={r.get('steps_logged')})")
print("\nRaw results dict:")
print(json.dumps(results, indent=2, default=str))

OUT = REPO_DIR / "data" / "pilot_pass4_results.json"
OUT.write_text(json.dumps(results, indent=2, default=str))
print(f"\nWrote {OUT}")
print("\nInterpretation guide:")
print("  - If G6 markedly lowers dead_step_frac vs G4 with similar coherence → adopt G=6.")
print("  - If T=1.1 further lowers dead_step_frac / raises entropy WITHOUT comp_len")
print("    ballooning or clip rate spiking → adopt T=1.1; else keep T=1.0.")
print("  - Pick the largest max_completion that didn't OOM (peak well under ~24GB).")
print("  - Then set difficulty sampling (launch_difficulty_pass3.sh) TEMPERATURE to the")
print("    chosen training T, and num_generations≈G, before sampling the curriculum.")
