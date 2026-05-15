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

INFERENCE BACKEND: vLLM. The first attempt used HF transformers generate to avoid
the numpy 2.x ABI conflict that `pip install vllm` causes in the container's shared
conda env -- but HF generate has no continuous batching, so 4464 long thinking-model
generations clocked in at ~93h. The fix is to run vLLM inside a *clean* `python -m venv`
(see scripts/launch_difficulty_v2.sh): vLLM brings its own torch 2.6 + numpy 2.x into
the isolated venv and never touches the container's pre-compiled scipy/sklearn. The
judger only needs sympy + antlr4 (pure Python), so nothing in this job needs scipy.

Runtime on DSMLP A5000 (24GB) with vLLM: ~2-4h for the full 1116 x 4 generations.

Smoke test first: set LIMIT=10 to run 10 prompts and eyeball throughput before
queueing the unrestricted job.

Usage (inside the clean venv -- see launch_difficulty_v2.sh):
    cd ~/151B_SP26_Competition
    LIMIT=10 HF_TOKEN=$(cat ~/.hf_token) python scripts/sample_difficulty_v2.py   # smoke test
    HF_TOKEN=$(cat ~/.hf_token) python scripts/sample_difficulty_v2.py            # full run
"""
import json, os, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ---- config ----
MODEL_ID       = "TrevorDuong/qwen3-4b-thinking-grpo-strict70"
NUM_SAMPLES    = 4
TEMPERATURE    = 1.0
TOP_P          = 0.95
TOP_K          = 20
MAX_NEW_TOKENS = 6144

# vLLM does continuous batching internally, so we hand it whole chunks of prompts
# and let the scheduler saturate the GPU. We still chunk (rather than one giant
# generate() call) so the JSONL is flushed periodically -> resumable if the 12h
# container times out mid-run.
CHUNK_PROMPTS  = 24          # 24 prompts x 4 samples = 96 seqs per call (was 48→OOM on A5000 logit sort)
MAX_MODEL_LEN  = 8192        # input p99 ~851 tok + 6144 gen, 8192 leaves headroom
GPU_MEM_UTIL   = 0.85        # reduced from 0.90; leaves headroom for logit sort on A5000 fallback
TENSOR_PARALLEL = 1          # DSMLP launches with -g 1. NOT 2.

# Smoke test: LIMIT=N caps the run to N prompts so you can eyeball throughput
# before committing to the full ~2-4h job.
LIMIT = int(os.environ.get("LIMIT", "0")) or None

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
from vllm import LLM, SamplingParams    # noqa: E402
from judger import Judger               # noqa: E402

LETTERS = "ABCDEFGHIJ"

# ---- helper functions ----
def normalize_gold(answer):
    if isinstance(answer, list):
        return [str(x) for x in answer]
    return [str(answer)]

def score_response(judger_inst, text, gold, options):
    gold_list = normalize_gold(gold)
    opts_per_gold = [options if options else []] * len(gold_list)
    try:
        return bool(judger_inst.auto_judge(pred=text, gold=gold_list, options=opts_per_gold))
    except Exception:
        return False

def build_prompt(tokenizer_inst, ex):
    is_mcq = bool(ex.get("options"))
    qt = ex["question"]
    if is_mcq:
        qt += "\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {v}" for i, v in enumerate(ex["options"]))
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_MCQ}] + FEWSHOT_MCQ + [{"role": "user", "content": qt}]
    else:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_MATH}] + FEWSHOT_MATH + [{"role": "user", "content": qt}]
    return tokenizer_inst.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# =====================================================================
# MAIN EXECUTION BLOCK (Protects against multiprocessing spawn errors)
# =====================================================================
if __name__ == "__main__":
    
    # Check CUDA status safely inside the main block
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}", flush=True)
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}  bf16={torch.cuda.is_bf16_supported()}", flush=True)

    # ---- load data ----
    with open(PUBLIC_PATH) as f:
        all_rows = [json.loads(l) for l in f]
    print(f"Loaded {len(all_rows)} prompts from public.jsonl", flush=True)

    # ---- resume: collect done ids, and assert schema of existing rows matches ----
    EXPECTED_KEYS = {"id", "is_mcq", "num_correct", "num_clipped", "n_samples", "samples"}
    EXPECTED_SAMPLE_KEYS = {"length", "clipped", "correct"}
    done_ids = set()
    if OUT_SAMPLES.exists():
        with open(OUT_SAMPLES) as f:
            for ln, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if set(r.keys()) != EXPECTED_KEYS:
                    sys.exit(f"SCHEMA DRIFT in {OUT_SAMPLES} line {ln+1}: keys={set(r.keys())} "
                             f"!= expected {EXPECTED_KEYS}. Aborting so resume can't mix schemas.")
                if r["samples"] and set(r["samples"][0].keys()) != EXPECTED_SAMPLE_KEYS:
                    sys.exit(f"SCHEMA DRIFT in {OUT_SAMPLES} line {ln+1}: sample keys mismatch. Aborting.")
                done_ids.add(r["id"])
        print(f"Resume: {len(done_ids)} prompts already sampled, skipping", flush=True)

    todo = [r for r in all_rows if r["id"] not in done_ids]
    if LIMIT:
        todo = todo[:LIMIT]
        print(f"LIMIT={LIMIT} -> smoke test on {len(todo)} prompts", flush=True)
    print(f"To sample: {len(todo)} prompts x {NUM_SAMPLES} = {len(todo)*NUM_SAMPLES} generations", flush=True)

    # ---- judger ----
    judger = Judger()

    if todo:
        hf_token = os.environ.get("HF_TOKEN")

        # ---- tokenizer (only for chat templating) ----
        print("Loading tokenizer ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)

        # ---- vLLM engine ----
        bf16 = torch.cuda.is_bf16_supported()
        dtype = "bfloat16" if bf16 else "float16"
        print(f"Loading vLLM engine for {MODEL_ID} (dtype={dtype}, tp={TENSOR_PARALLEL}) ...", flush=True)
        llm = LLM(
            model=MODEL_ID,
            dtype=dtype,
            tensor_parallel_size=TENSOR_PARALLEL,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            n=NUM_SAMPLES,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_tokens=MAX_NEW_TOKENS,
        )
        print("vLLM engine ready.", flush=True)

        # ---- sample and score (chunked, streaming write for resumability) ----
        written = 0
        t0 = time.time()
        with open(OUT_SAMPLES, "a") as out:
            for i in range(0, len(todo), CHUNK_PROMPTS):
                chunk = todo[i:i + CHUNK_PROMPTS]
                prompts = [build_prompt(tokenizer, ex) for ex in chunk]
                t_chunk = time.time()
                outputs = llm.generate(prompts, sampling_params)
                chunk_secs = time.time() - t_chunk

                for ex, output in zip(chunk, outputs):
                    is_mcq = bool(ex.get("options"))
                    samples = []
                    num_correct = 0
                    num_clipped = 0
                    for comp in output.outputs:
                        text = comp.text
                        length = len(comp.token_ids)
                        clipped = (comp.finish_reason == "length")
                        ok = score_response(judger, text, ex["answer"], ex.get("options"))
                        samples.append({"length": length, "clipped": clipped, "correct": ok})
                        if ok:      num_correct += 1
                        if clipped: num_clipped += 1
                    out.write(json.dumps({
                        "id": ex["id"],
                        "is_mcq": is_mcq,
                        "num_correct": num_correct,
                        "num_clipped": num_clipped,
                        "n_samples": NUM_SAMPLES,
                        "samples": samples,
                    }) + "\n")
                    written += 1

                out.flush()
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                eta = (len(todo) - written) / rate / 60 if rate > 0 else float("inf")
                print(f"  [{written}/{len(todo)}] chunk={chunk_secs/60:.1f}m  "
                      f"elapsed={elapsed/60:.1f}m  rate={rate*60:.1f}/min  ETA={eta:.1f}m", flush=True)

        print(f"\nSampling complete. {written} new rows written to {OUT_SAMPLES}", flush=True)

    if LIMIT:
        print(f"\nLIMIT={LIMIT} smoke test done -- not building curriculum. "
              f"Re-run without LIMIT for the full job.", flush=True)
        sys.exit(0)

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