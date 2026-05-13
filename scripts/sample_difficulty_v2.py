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

INFERENCE BACKEND: HF transformers generate, NOT vLLM. The scipy-ml-notebook:stable
container has torch 2.5 + numpy 1.x baked in; upgrading torch to 2.6 (vllm 0.7+
requirement for Qwen3 support) pulls numpy 2.x, which breaks the pre-compiled
scipy/sklearn used by transformers. HF generate works out of the box.

Runtime on DSMLP A5000 (24GB) with batch_size=2, ~2.5h with GQA, ~5h worst case.

Usage (inside the container):
    cd ~/151B_SP26_Competition
    pip install -q --user sympy "antlr4-python3-runtime==4.11"
    HF_TOKEN=$(cat ~/.hf_token) python scripts/sample_difficulty_v2.py
"""
import json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}", flush=True)
print(f"torch={torch.__version__}  cuda={torch.version.cuda}  bf16={torch.cuda.is_bf16_supported()}", flush=True)

# ---- config ----
MODEL_ID       = "TrevorDuong/qwen3-4b-thinking-grpo-strict70"
NUM_SAMPLES    = 4
TEMPERATURE    = 1.0
TOP_P          = 0.95
TOP_K          = 20
MAX_NEW_TOKENS = 6144
# batch_size = number of prompts per generate() call.
# Total in-flight sequences = BATCH_PROMPTS * NUM_SAMPLES.
# Qwen3-4B has GQA-8, so KV is small; batch=2 -> 8 seqs -> ~5GB KV at 6144 tok.
BATCH_PROMPTS  = 2

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

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
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

# ---- load model + tokenizer ----
hf_token = os.environ.get("HF_TOKEN")

print(f"Loading tokenizer ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
# Left-pad so generation continues from the prompt's right edge for all batch members
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if todo:
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"Loading model {MODEL_ID} in {dtype} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    model.eval()
    print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    gen_kwargs = dict(
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=NUM_SAMPLES,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

# ---- build chat-templated prompts ----
def build_prompt(ex):
    is_mcq = bool(ex.get("options"))
    qt = ex["question"]
    if is_mcq:
        qt += "\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {v}" for i, v in enumerate(ex["options"]))
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_MCQ}] + FEWSHOT_MCQ + [{"role": "user", "content": qt}]
    else:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_MATH}] + FEWSHOT_MATH + [{"role": "user", "content": qt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# ---- judger ----
judger = Judger()

def normalize_gold(answer):
    if isinstance(answer, list):
        return [str(x) for x in answer]
    return [str(answer)]

def score_response(text, gold, options):
    gold_list = normalize_gold(gold)
    opts_per_gold = [options if options else []] * len(gold_list)
    try:
        return bool(judger.auto_judge(pred=text, gold=gold_list, options=opts_per_gold))
    except Exception:
        return False

# ---- sample and score (streaming write for resumability) ----
import time
written = 0
t0 = time.time()

with open(OUT_SAMPLES, "a") as out:
    for i in range(0, len(todo), BATCH_PROMPTS):
        batch_rows = todo[i:i + BATCH_PROMPTS]
        batch_prompts = [build_prompt(ex) for ex in batch_rows]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
        input_len = enc.input_ids.shape[1]

        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kwargs)
        # out_ids: (BATCH * NUM_SAMPLES, input_len + max_new)
        # Reshape to (BATCH, NUM_SAMPLES, seq_len)
        seq_len = out_ids.shape[1]
        out_ids = out_ids.view(len(batch_rows), NUM_SAMPLES, seq_len)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        for j, ex in enumerate(batch_rows):
            is_mcq = bool(ex.get("options"))
            samples = []
            num_correct = 0
            num_clipped = 0
            for k in range(NUM_SAMPLES):
                gen_ids = out_ids[j, k, input_len:].tolist()
                # Find end: first EOS or first pad
                end = len(gen_ids)
                for idx, t in enumerate(gen_ids):
                    if t == eos_id or t == pad_id:
                        end = idx
                        break
                gen_ids = gen_ids[:end]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                length = len(gen_ids)
                # Clipped iff hit max_new_tokens without EOS (and didn't end on pad)
                clipped = (length >= MAX_NEW_TOKENS)
                ok = score_response(text, ex["answer"], ex.get("options"))
                samples.append({
                    "length": length,
                    "clipped": clipped,
                    "correct": ok,
                })
                if ok:      num_correct += 1
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
        elapsed = time.time() - t0
        rate = written / elapsed if elapsed > 0 else 0
        eta = (len(todo) - written) / rate / 60 if rate > 0 else float("inf")
        print(f"  [{written}/{len(todo)}] elapsed={elapsed/60:.1f}m  rate={rate*60:.1f}/min  ETA={eta:.1f}m", flush=True)

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
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    }, f, indent=2)

print(f"\nWrote curriculum: {OUT_CURRICULUM}")
print("Done.")
