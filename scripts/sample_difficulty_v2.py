"""Difficulty sampling v2: re-sample the current GRPO model on all 1126 public.jsonl
prompts to build a fresh curriculum for future GRPO training.

The strict-70 curriculum (sweet_spot_ids_clean.json) was sampled from base
Qwen3-4B-Thinking-2507. After exp_009 GRPO training, some prompts have shifted:
  - Previously hard (1-3/4) prompts may now be solved 4/4 -> no gradient signal
  - Previously easy (4/4) prompts may have regressed to 1-3/4 -> new training targets

Output:
  data/difficulty_samples_v2.jsonl  -- one row per prompt with per-sample stats
  data/sweet_spot_v2_ids.json       -- filtered IDs: 1<=num_correct<=3 AND no clipping

Resumable: skip prompts already present in difficulty_samples_v2.jsonl.

Run on DSMLP A5000 (24GB). Estimated runtime ~1.5h with vLLM, ~5-6h with HF.

Usage (inside the container):
    cd ~/151B_SP26_Competition
    pip install -q torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124
    pip install -q vllm==0.8.5 sympy antlr4-python3-runtime==4.11
    HF_TOKEN=$(cat ~/.hf_token) python scripts/sample_difficulty_v2.py
"""
import json, os, sys, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}", flush=True)
print(f"torch={torch.__version__}  cuda={torch.version.cuda}  bf16={torch.cuda.is_bf16_supported()}", flush=True)

# ---- config ----
MODEL_ID       = "TrevorDuong/qwen3-4b-thinking-grpo"
NUM_SAMPLES    = 4
TEMPERATURE    = 1.0
TOP_P          = 0.95
TOP_K          = 20
MAX_TOKENS     = 6144
MAX_MODEL_LEN  = 8192   # 6144 generation + prompt headroom (input p99 ~1000)
GPU_MEM_UTIL   = 0.90
CHUNK          = 80     # prompts per vLLM batch; 80*4=320 generations in flight

PUBLIC_PATH    = REPO / "data" / "public.jsonl"
OUT_SAMPLES    = REPO / "data" / "difficulty_samples_v2.jsonl"
OUT_CURRICULUM = REPO / "data" / "sweet_spot_v2_ids.json"

# Reuse exp_009's training-time prompts. The GRPO model expects these exact prompts;
# sampling with different prompts puts it out-of-distribution and gives misleading
# difficulty estimates.
EXP_009_DIR = REPO / "experiments" / "exp_009_grpo"
sys.path.insert(0, str(EXP_009_DIR))
from prompts import (  # noqa: E402
    SYSTEM_PROMPT_MATH, SYSTEM_PROMPT_MCQ,
    FEWSHOT_MATH, FEWSHOT_MCQ,
)

from transformers import AutoTokenizer  # noqa: E402
from judger import Judger  # noqa: E402

LETTERS = "ABCDEFGHIJ"

# ---- load data ----
with open(PUBLIC_PATH) as f:
    all_rows = [json.loads(l) for l in f]
print(f"Loaded {len(all_rows)} prompts from public.jsonl", flush=True)

done_ids = set()
if OUT_SAMPLES.exists():
    with open(OUT_SAMPLES) as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"Resume: {len(done_ids)} prompts already sampled, skipping", flush=True)

todo = [r for r in all_rows if r["id"] not in done_ids]
print(f"To sample: {len(todo)} prompts x {NUM_SAMPLES} = {len(todo)*NUM_SAMPLES} generations", flush=True)
if not todo:
    print("Nothing to sample. Jumping to curriculum build step.")

# ---- build chat-templated prompts ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
prompts_text = []
for ex in todo:
    is_mcq = bool(ex.get("options"))
    qt = ex["question"]
    if is_mcq:
        qt += "\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {v}" for i, v in enumerate(ex["options"]))
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_MCQ}] + FEWSHOT_MCQ + [{"role": "user", "content": qt}]
    else:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_MATH}] + FEWSHOT_MATH + [{"role": "user", "content": qt}]
    prompts_text.append(
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    )

# ---- load vLLM ----
if todo:
    from vllm import LLM, SamplingParams
    bf16 = torch.cuda.is_bf16_supported()
    dtype = "bfloat16" if bf16 else "float16"
    print(f"Loading {MODEL_ID} via vLLM (dtype={dtype}, max_model_len={MAX_MODEL_LEN})...", flush=True)
    llm = LLM(
        model=MODEL_ID,
        dtype=dtype,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        n=NUM_SAMPLES,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
    )

# ---- judger ----
judger = Judger()

def normalize_gold(answer):
    if isinstance(answer, list):
        return [str(x) for x in answer]
    return [str(answer)]

def score_response(text, gold, options):
    """Return True iff judger says the response's boxed answer matches gold."""
    gold_list = normalize_gold(gold)
    opts_per_gold = [options if options else []] * len(gold_list)
    try:
        return bool(judger.auto_judge(pred=text, gold=gold_list, options=opts_per_gold))
    except Exception:
        return False

# ---- sample and score (streaming write for resumability) ----
written = 0
with open(OUT_SAMPLES, "a") as out:
    for i in range(0, len(todo), CHUNK):
        batch_rows = todo[i:i + CHUNK]
        batch_prompts = prompts_text[i:i + CHUNK]
        outputs = llm.generate(batch_prompts, sampling)
        for ex, out_obj in zip(batch_rows, outputs):
            is_mcq = bool(ex.get("options"))
            samples = []
            num_correct = 0
            num_clipped = 0
            for o in out_obj.outputs:
                text = o.text
                length = len(o.token_ids)
                clipped = (o.finish_reason == "length")
                ok = score_response(text, ex["answer"], ex.get("options"))
                samples.append({
                    "length": length,
                    "clipped": clipped,
                    "correct": ok,
                    "finish_reason": o.finish_reason,
                })
                if ok: num_correct += 1
                if clipped: num_clipped += 1
            row = {
                "id": ex["id"],
                "is_mcq": is_mcq,
                "num_correct": num_correct,
                "num_clipped": num_clipped,
                "n_samples": NUM_SAMPLES,
                "samples": samples,
            }
            out.write(json.dumps(row) + "\n")
            written += 1
        out.flush()
        print(f"  Saved {written}/{len(todo)} prompts so far", flush=True)

print(f"\nSampling complete. {written} new rows written to {OUT_SAMPLES}", flush=True)

# ---- build sweet-spot v2 curriculum ----
print("\nBuilding curriculum (strict criterion: 1<=num_correct<=3 AND num_clipped==0)...")
sweet, all_stats, id_is_mcq = [], [], {}
with open(OUT_SAMPLES) as f:
    for line in f:
        r = json.loads(line)
        all_stats.append(r)
        id_is_mcq[r["id"]] = r["is_mcq"]
        if 1 <= r["num_correct"] <= 3 and r["num_clipped"] == 0:
            sweet.append(r["id"])

mcq_sweet = sum(1 for sid in sweet if id_is_mcq[sid])
ff_sweet  = len(sweet) - mcq_sweet
print(f"Sweet-spot v2: {len(sweet)} prompts ({mcq_sweet} MCQ, {ff_sweet} free-form)")

# Distribution summary for sanity
dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
clipped_count = 0
for r in all_stats:
    dist[r["num_correct"]] = dist.get(r["num_correct"], 0) + 1
    if r["num_clipped"] > 0:
        clipped_count += 1
print(f"num_correct distribution: 0={dist[0]}  1={dist[1]}  2={dist[2]}  3={dist[3]}  4={dist[4]}")
print(f"Prompts with >=1 clipped sample: {clipped_count}/{len(all_stats)}")

with open(OUT_CURRICULUM, "w") as f:
    json.dump({
        "sweet_ids": sorted(sweet),
        "n_total": len(all_stats),
        "n_sweet": len(sweet),
        "breakdown": {"mcq": mcq_sweet, "ff": ff_sweet},
        "num_correct_distribution": dist,
        "model": MODEL_ID,
        "num_samples": NUM_SAMPLES,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }, f, indent=2)

print(f"\nWrote curriculum: {OUT_CURRICULUM}")
print(f"Done.")
