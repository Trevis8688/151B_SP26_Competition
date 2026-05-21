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
import atexit, json, os, re, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
import torch


# SymPy hang defense (replaces an earlier SIGALRM attempt at commit 3f5817a that did
# NOT fire -- parse_latex runs Antlr's C/Cython runtime without yielding to Python
# bytecode, so the signal sat pending until return and the run wedged again at the
# same chunk it had previously). Under the matched-sampler config (top_k=-1, top_p=1.0,
# T=1.0) the policy occasionally emits pathological \boxed{} contents -- e.g. runaway
# repetition of the same token, or deeply nested LaTeX -- that parse_latex cannot
# parse and from which it cannot return. Two layers:
#   1) MAX_BOXED_LEN -- in-process pre-filter on \boxed{} content length. Real math
#      and MCQ answers are < 50 chars; > 300 is overwhelmingly runaway garbage. These
#      short-circuit to wrong without touching SymPy.
#   2) JUDGE_TIMEOUT_SECONDS -- any auto_judge that survives pre-filter runs in a
#      one-worker subprocess pool we hard-kill on timeout (concurrent.futures pool
#      shutdown). This works for C-extension hangs because we terminate the worker
#      *process*, not the C call.
JUDGE_TIMEOUT_SECONDS = int(os.environ.get("JUDGE_TIMEOUT_SECONDS", "15"))
MAX_BOXED_LEN         = int(os.environ.get("MAX_BOXED_LEN", "300"))

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ---- config ----
# Both overridable via env so the same script can resample later policies
# (e.g. the exp_015 pass-2 model) without duplicating the file. Defaults
# preserve the original exp_015-prep behaviour exactly.
MODEL_ID       = os.environ.get("MODEL_ID", "TrevorDuong/qwen3-4b-thinking-grpo-strict70")
OUT_SUFFIX     = os.environ.get("OUT_SUFFIX", "v2")  # difficulty_samples_<suffix>.jsonl
# Sampling params are env-overridable so the curriculum can be measured under the
# SAME per-token distribution that GRPO training will use (otherwise the difficulty
# band measured here doesn't predict training-time reward variance). NOTE the
# disable convention differs by backend: vLLM (here) disables top_k with TOP_K=-1
# and top_p with TOP_P=1.0; TRL/transformers (training) disables top_k with top_k=0.
# So "match training top_k=0" means passing TOP_K=-1 to this script.
NUM_SAMPLES    = int(os.environ.get("NUM_SAMPLES", "4"))
TEMPERATURE    = float(os.environ.get("TEMPERATURE", "1.0"))
TOP_P          = float(os.environ.get("TOP_P", "0.95"))
TOP_K          = int(os.environ.get("TOP_K", "20"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "6144"))

# vLLM does continuous batching internally, so we hand it whole chunks of prompts
# and let the scheduler saturate the GPU. We still chunk (rather than one giant
# generate() call) so the JSONL is flushed periodically -> resumable if the 12h
# container times out mid-run.
CHUNK_PROMPTS  = int(os.environ.get("CHUNK_PROMPTS", "24"))  # keep CHUNK_PROMPTS*NUM_SAMPLES ~96 seqs/call (48*4 OOM'd on A5000 logit sort); for NUM_SAMPLES=8 use CHUNK_PROMPTS=12
MAX_MODEL_LEN  = 8192        # input p99 ~851 tok + 6144 gen, 8192 leaves headroom
GPU_MEM_UTIL   = 0.85        # reduced from 0.90; leaves headroom for logit sort on A5000 fallback
TENSOR_PARALLEL = 1          # DSMLP launches with -g 1. NOT 2.

# Smoke test: LIMIT=N caps the run to N prompts so you can eyeball throughput
# before committing to the full ~2-4h job.
LIMIT = int(os.environ.get("LIMIT", "0")) or None

PUBLIC_PATH    = REPO / "data" / "public.jsonl"
OUT_SAMPLES    = REPO / "data" / f"difficulty_samples_{OUT_SUFFIX}.jsonl"
OUT_CURRICULUM = REPO / "data" / f"sweet_spot_{OUT_SUFFIX}_ids.json"

# Reuse exp_009's training-time prompts. The GRPO model expects these exact prompts;
# sampling with different prompts puts it out-of-distribution and gives misleading
# difficulty estimates.
EXP_009_DIR = REPO / "experiments" / "exp_009_grpo"
sys.path.insert(0, str(EXP_009_DIR))
from prompts import (  # noqa: E402
    SYSTEM_PROMPT_MATH, SYSTEM_PROMPT_MCQ,
    FEWSHOT_MATH, FEWSHOT_MCQ,
)

LETTERS = "ABCDEFGHIJ"

# ---- helper functions ----
def normalize_gold(answer):
    if isinstance(answer, list):
        return [str(x) for x in answer]
    return [str(answer)]


# Pre-filter helper: extract the LAST \boxed{...} content from a response, honoring
# brace nesting. Returns None on missing-boxed or unbalanced-braces -- both treated
# as poison (cannot evaluate -> wrong).
_BOXED_OPEN_RE = re.compile(r"\\boxed\{")
def _last_boxed_content(text):
    matches = list(_BOXED_OPEN_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    n = len(text)
    while i < n and depth > 0:
        c = text[i]
        if c == "\\" and i + 1 < n:
            i += 2  # skip escaped char (e.g. \{ \})
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None  # unbalanced

# ---- subprocess judge worker (spawn ctx, distinct from vLLM's mp infra) ----
# Worker globals (only touched inside the worker process; main proc leaves them None).
_worker_judger = None
def _worker_init():
    """Run once per worker process. Imports Judger fresh inside the worker."""
    global _worker_judger
    from judger import Judger
    _worker_judger = Judger()

def _judge_call(text, gold_list, opts_per_gold):
    """Run inside the worker. _worker_judger is set by _worker_init at pool startup."""
    return bool(_worker_judger.auto_judge(pred=text, gold=gold_list, options=opts_per_gold))

_SPAWN_CTX = mp.get_context("spawn")
_pool = None
def _get_pool():
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=1, mp_context=_SPAWN_CTX,
                                    initializer=_worker_init)
    return _pool

def _reset_pool():
    """Hard-kill the current pool. The next _get_pool() spins up a fresh worker."""
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None

atexit.register(_reset_pool)


def score_response(text, gold, options):
    """Score one response. Two-layer hang defense (see comment block at top)."""
    gold_list = normalize_gold(gold)
    opts_per_gold = [options if options else []] * len(gold_list)

    # Layer 1: in-process pre-filter on extracted \boxed{} content.
    boxed = _last_boxed_content(text)
    if boxed is None or len(boxed) > MAX_BOXED_LEN:
        return False

    # Layer 2: subprocess timeout. The worker can be hard-killed via pool shutdown
    # regardless of where SymPy is stuck (handles C-extension hangs that SIGALRM cannot).
    try:
        pool = _get_pool()
        fut = pool.submit(_judge_call, text, gold_list, opts_per_gold)
        return fut.result(timeout=JUDGE_TIMEOUT_SECONDS)
    except FuturesTimeout:
        _reset_pool()  # worker wedged in C code; throw it away
        return False
    except Exception:
        _reset_pool()  # any other pool error -- play it safe, force a fresh worker
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
    # Heavy imports live INSIDE __main__ so spawn'd judge workers re-importing this
    # module don't pay the vLLM / transformers import cost (vLLM init alone is ~30s
    # and would dwarf the per-call judging budget). The worker only needs `judger`,
    # which it imports itself inside _worker_init.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

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

    # Judger lives in the subprocess worker (see _worker_init); the main process never
    # calls auto_judge directly. This keeps SymPy hangs out of the main loop.

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
                        ok = score_response(text, ex["answer"], ex.get("options"))
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
            "top_p": TOP_P,
            "top_k": TOP_K,
        }, f, indent=2)

    print(f"\nWrote curriculum: {OUT_CURRICULUM}")
    print("Done.")