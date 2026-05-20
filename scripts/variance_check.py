"""Variance check: measure run-to-run noise of the exp_018 stage-1 inference.

Within a single vLLM call, n=NUM_SAMPLES gives N independent decode trajectories
from the same prefilled prompt -- statistically equivalent to running Kaggle
inference N times with different RNG seeds.

Sampling defaults match exp_018 stage-1 (Kaggle T=0.6, top_p=0.95, top_k=20,
max_tokens=8192). The model defaults to the pass-2 GRPO policy that exp_018
used. The prompts come from exp_009 (same prompt chain used by exp_017/018).

Three modes (default = generate):
  GENERATE (no flag):       write per-prompt samples to OUT
  --summarize-diff PATH:    print fraction of prompts whose samples disagreed
                            on the \\boxed answer (cheap smoke metric)
  --summarize-score PATH:   per-sample accuracy + std across samples (the
                            actual variance estimate)

Both summaries can be re-run on a saved OUT file without regenerating.

Usage (inside the venv built by variance_check.sh):
    LIMIT=100 NUM_SAMPLES=2 OUT=smoke.jsonl python scripts/variance_check.py
    python scripts/variance_check.py --summarize-diff smoke.jsonl
    LIMIT=0   NUM_SAMPLES=3 OUT=full.jsonl  python scripts/variance_check.py
    python scripts/variance_check.py --summarize-score full.jsonl
"""
import argparse, json, os, re, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PUBLIC_PATH = REPO / "data" / "public.jsonl"

# Defaults match exp_018 stage-1 (= exp_017 config = Kaggle settings).
MODEL_ID       = os.environ.get("MODEL_ID", "TrevorDuong/qwen3-4b-thinking-grpo-pass2")
NUM_SAMPLES    = int(os.environ.get("NUM_SAMPLES", "3"))
TEMPERATURE    = float(os.environ.get("TEMPERATURE", "0.6"))
TOP_P          = float(os.environ.get("TOP_P", "0.95"))
TOP_K          = int(os.environ.get("TOP_K", "20"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "8192"))
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", "10240"))
# Conservative chunk: at max_tokens=8192 each sequence is bigger than the
# difficulty-sampling job (which used 6144). CHUNK*N <= 48 keeps logit-sort
# well under the A5000 24GB OOM line.
CHUNK_PROMPTS  = int(os.environ.get("CHUNK_PROMPTS", "16"))
LIMIT          = int(os.environ.get("LIMIT", "0")) or None
OUT            = Path(os.environ.get("OUT", str(REPO / "data" / "variance_check.jsonl")))
GPU_MEM_UTIL   = float(os.environ.get("GPU_MEM_UTIL", "0.85"))

LETTERS = "ABCDEFGHIJ"

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text):
    if not text:
        return None
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


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


def summarize_diff(out_path):
    n, diff = 0, 0
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            boxes = [s.get("boxed") for s in r["samples"]]
            if len(set(boxes)) > 1:
                diff += 1
            n += 1
    frac = diff / n if n else 0.0
    # Easy-to-grep summary line (used by the launcher to decide Phase B).
    print(f"VARIANCE_CHECK_SUMMARY_DIFF: prompts={n} diff_boxed={diff} diff_fraction={frac:.4f}")
    return frac


def summarize_score(out_path):
    per_sample = None
    n = 0
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            samples = r["samples"]
            if per_sample is None:
                per_sample = [[] for _ in range(len(samples))]
            for i, s in enumerate(samples):
                per_sample[i].append(1 if s["correct"] else 0)
            n += 1
    if per_sample is None or len(per_sample) == 0:
        print("VARIANCE_CHECK_SUMMARY_SCORE: no samples")
        return None, None
    scores = [sum(s) / max(1, len(s)) for s in per_sample]
    mean = sum(scores) / len(scores)
    if len(scores) > 1:
        var = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
    else:
        var = 0.0
    std = var ** 0.5
    print(f"VARIANCE_CHECK_SUMMARY_SCORE: prompts={n} n_samples={len(per_sample)} "
          f"mean={mean:.4f} std={std:.4f}")
    for i, s in enumerate(scores):
        print(f"  sample[{i}] = {s*100:.2f}% ({sum(per_sample[i])}/{n})")
    print(f"  mean across samples = {mean*100:.2f}%")
    print(f"  std  across samples = {std*100:.4f}pp")
    # Aggregate std scales ~1/sqrt(N_questions). Translate from public(1126) to private(943).
    kaggle_std = std * (1126.0 / 943.0) ** 0.5
    print(f"  estimated Kaggle σ (scaled to 943 private q): {kaggle_std*100:.4f}pp")
    print(f"  approx 95% CI on a single Kaggle submission: ±{2*kaggle_std*100:.2f}pp")
    return mean, std


def generate():
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}", flush=True)
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}  bf16={torch.cuda.is_bf16_supported()}", flush=True)

    with open(PUBLIC_PATH) as f:
        all_rows = [json.loads(l) for l in f]
    print(f"Loaded {len(all_rows)} prompts from public.jsonl", flush=True)

    rows = all_rows[:LIMIT] if LIMIT else all_rows
    print(f"To sample: {len(rows)} prompts x {NUM_SAMPLES} = {len(rows)*NUM_SAMPLES} generations", flush=True)
    print(f"Sampling: model={MODEL_ID} T={TEMPERATURE} top_p={TOP_P} top_k={TOP_K} "
          f"max_tokens={MAX_NEW_TOKENS} chunk={CHUNK_PROMPTS}", flush=True)

    # Reuse exp_009 prompts — exp_017/018 stage-1 used these (same as the GRPO model's training prompts).
    sys.path.insert(0, str(REPO / "experiments" / "exp_009_grpo"))
    from prompts import (
        SYSTEM_PROMPT_MATH, SYSTEM_PROMPT_MCQ,
        FEWSHOT_MATH, FEWSHOT_MCQ,
    )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from judger import Judger

    judger = Judger()
    hf_token = os.environ.get("HF_TOKEN")

    print("Loading tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)

    bf16 = torch.cuda.is_bf16_supported()
    dtype = "bfloat16" if bf16 else "float16"
    print(f"Loading vLLM for {MODEL_ID} (dtype={dtype}) ...", flush=True)
    llm = LLM(
        model=MODEL_ID,
        dtype=dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        n=NUM_SAMPLES,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_NEW_TOKENS,
    )
    print("vLLM engine ready.", flush=True)

    def build_prompt(ex):
        is_mcq = bool(ex.get("options"))
        qt = ex["question"]
        if is_mcq:
            qt += "\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {v}" for i, v in enumerate(ex["options"]))
            msgs = [{"role": "system", "content": SYSTEM_PROMPT_MCQ}] + FEWSHOT_MCQ + [{"role": "user", "content": qt}]
        else:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT_MATH}] + FEWSHOT_MATH + [{"role": "user", "content": qt}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()  # variance check is one-shot; clean slate each run

    written = 0
    t0 = time.time()
    with open(OUT, "a") as out:
        for i in range(0, len(rows), CHUNK_PROMPTS):
            chunk = rows[i:i + CHUNK_PROMPTS]
            prompts = [build_prompt(ex) for ex in chunk]
            t_chunk = time.time()
            outputs = llm.generate(prompts, sp)
            chunk_secs = time.time() - t_chunk

            for ex, output in zip(chunk, outputs):
                samples = []
                for comp in output.outputs:
                    text = comp.text
                    boxed = extract_boxed(text)
                    correct = score_response(judger, text, ex["answer"], ex.get("options"))
                    samples.append({
                        "boxed": boxed,
                        "correct": correct,
                        "clipped": (comp.finish_reason == "length"),
                        "length": len(comp.token_ids),
                    })
                out.write(json.dumps({
                    "id": ex["id"],
                    "is_mcq": bool(ex.get("options")),
                    "samples": samples,
                }) + "\n")
                written += 1
            out.flush()
            elapsed = time.time() - t0
            rate = written / elapsed if elapsed > 0 else 0
            eta = (len(rows) - written) / rate / 60 if rate > 0 else float("inf")
            print(f"  [{written}/{len(rows)}] chunk={chunk_secs/60:.1f}m  "
                  f"elapsed={elapsed/60:.1f}m  rate={rate*60:.1f}/min  ETA={eta:.1f}m", flush=True)

    print(f"\nSampling complete. {written} prompts -> {OUT}", flush=True)
    print("--- summaries ---", flush=True)
    summarize_diff(OUT)
    if (LIMIT or len(rows)) >= 200:
        summarize_score(OUT)
    else:
        print("(skipping score summary: too few prompts for a meaningful std)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-diff", type=str, default=None)
    parser.add_argument("--summarize-score", type=str, default=None)
    args = parser.parse_args()
    if args.summarize_diff:
        summarize_diff(args.summarize_diff)
        return
    if args.summarize_score:
        summarize_score(args.summarize_score)
        return
    generate()


if __name__ == "__main__":
    main()
