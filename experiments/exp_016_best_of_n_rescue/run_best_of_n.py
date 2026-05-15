"""exp_016 — best-of-N self-consistency runner.

Runs N stochastic samples per candidate question through the same GRPO model
used by exp_014 stage 1, clusters the boxed answers by Judger.auto_judge
pairwise equivalence, and picks the largest cluster's representative if it has
>= min_cluster_size members. Otherwise falls back to exp_014's answer (no
regression risk).

Modes:
  --target public  : eval on public.jsonl. Candidates = exp_014 wrong cases
                     (wrong_ff with boxed + missing_boxed + wrong_mcq).
                     Requires gold answers, used for measurement only.
  --target private : produce submission. Candidates depend on --private-mode:
      --private-mode full    : all 943 private questions (default; ~3.2h)
      --private-mode missing : only IDs where exp_014 lacks \\boxed (~20 min)

Outputs (in --out-dir, default /kaggle/working when on Kaggle, else cwd):
  public_responses.jsonl   (merged exp_014 + voted overrides for public)
  private_responses.jsonl  (merged for private; only written for --target private)
  submission.csv           (from private_responses.jsonl)
  best_of_n_stats.json     (per-candidate vote breakdown, cluster sizes,
                            flip counts, fallback counts)

Local smoke test:
  LIMIT=5 python experiments/exp_016_best_of_n_rescue/run_best_of_n.py \\
      --target public

Kaggle: see best_of_n_notebook.ipynb in the same folder — it sets vLLM env
vars + cuda stub path, then imports and calls main().
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EXP_DIR))

# Judger lives at repo root in dev; also bundled into the utils dataset on Kaggle.
def _find_judger():
    import glob
    for p in [REPO / "judger.py",
              *map(Path, glob.glob("/kaggle/input/**/judger.py", recursive=True))]:
        if p.exists():
            return p
    raise FileNotFoundError("judger.py not located")

_judger_path = _find_judger()
if _judger_path.parent != REPO:
    sys.path.insert(0, str(_judger_path.parent))
from judger import Judger  # noqa: E402

LETTERS = "ABCDEFGHIJ"

# Match the same pattern exp_010 reward uses for finding boxed in completions.
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_post_think(text: str) -> str:
    idx = text.rfind("</think>")
    return text if idx == -1 else text[idx + len("</think>"):]


def extract_boxed(response: str) -> str | None:
    """Return the last \\boxed{...} content from `response`, preferring the
    portion after </think>. None if no boxed found."""
    if not response:
        return None
    post = extract_post_think(response)
    matches = _BOXED_RE.findall(post) or _BOXED_RE.findall(response)
    if not matches:
        return None
    return matches[-1].strip()


def cluster_by_judger(answers: list[str], gold_count_hint: int, options: list,
                      judger: Judger) -> list[list[int]]:
    """Cluster `answers` (list of raw \\boxed contents) into equivalence classes
    by Judger.auto_judge. Returns a list of clusters, each a list of indices
    into `answers`.

    We treat answer `a` and `b` as equivalent if running auto_judge with `a` as
    the prediction and `b` (wrapped as gold list) as gold returns True. The
    judger normalizes both sides, so 325*326 ≡ 105950 succeeds.
    """
    clusters: list[list[int]] = []
    for i, a in enumerate(answers):
        if a is None:
            clusters.append([i])
            continue
        # Wrap `a` as if it were a response to judge against each existing
        # cluster's representative. The judger expects pred to look like a model
        # response, so we re-embed in \boxed{}.
        pred = f"\\boxed{{{a}}}"
        placed = False
        for c in clusters:
            rep_idx = c[0]
            rep = answers[rep_idx]
            if rep is None:
                continue
            gold = _split_for_judge(rep, gold_count_hint)
            opts_list = [options if options else None] * len(gold)
            try:
                ok = judger.auto_judge(pred=pred, gold=gold, options=opts_list)
            except Exception:
                ok = False
            if ok:
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def _split_for_judge(raw_boxed_content: str, gold_count_hint: int) -> list[str]:
    """When gold is a list of N parts, the judger expects a list of length N.
    If the raw boxed content has comma-separated values matching N, split it;
    otherwise treat as a single value (the judger handles strings of any form).
    """
    if gold_count_hint <= 1:
        return [raw_boxed_content]
    parts = [p.strip() for p in raw_boxed_content.split(",") if p.strip()]
    if len(parts) == gold_count_hint:
        return parts
    return [raw_boxed_content] * gold_count_hint


def build_chat(question: str, options: list | None, prompts_module) -> list[dict]:
    """Build the chat messages for the GRPO model — same shape as exp_009 used."""
    is_mcq = bool(options)
    if is_mcq:
        opts_block = "\n".join(f"{LETTERS[i]}. {v}" for i, v in enumerate(options))
        user = f"{question}\n\nOptions:\n{opts_block}"
        msgs = [{"role": "system", "content": prompts_module.SYSTEM_PROMPT_MCQ}]
        msgs.extend(prompts_module.FEWSHOT_MCQ)
    else:
        user = question
        msgs = [{"role": "system", "content": prompts_module.SYSTEM_PROMPT_MATH}]
        msgs.extend(prompts_module.FEWSHOT_MATH)
    msgs.append({"role": "user", "content": user})
    return msgs


# ============================================================
# Candidate selection
# ============================================================
def select_public_candidates(public_qs: dict, exp14_responses: dict,
                              judger: Judger) -> tuple[list[int], list[int]]:
    """Returns (default_budget_ids, big_budget_ids).

    default_budget_ids: candidates that run at max_tokens_default (wrong_ff + wrong_mcq)
    big_budget_ids:     missing_boxed candidates run at max_tokens_missing_boxed
    """
    default_ids = []
    big_ids = []
    for qid, q in public_qs.items():
        resp = exp14_responses.get(qid)
        if resp is None:
            continue
        text = resp["response"]
        is_mcq = bool(q.get("options"))
        # missing_boxed first — distinct token budget
        if "\\boxed" not in text:
            big_ids.append(qid)
            continue
        # judge against gold
        gold = q["answer"] if isinstance(q["answer"], list) else [q["answer"]]
        opts_list = [q.get("options") or None] * len(gold)
        try:
            correct = judger.auto_judge(pred=text, gold=[str(g) for g in gold],
                                         options=opts_list)
        except Exception:
            correct = False
        if not correct:
            default_ids.append(qid)
    return sorted(default_ids), sorted(big_ids)


def select_private_candidates(private_qs: dict, exp14_responses: dict,
                                mode: str) -> tuple[list[int], list[int]]:
    if mode == "full":
        default_ids = []
        big_ids = []
        for qid in private_qs:
            resp = exp14_responses.get(qid)
            if resp is None or "\\boxed" not in (resp.get("response") or ""):
                big_ids.append(qid)
            else:
                default_ids.append(qid)
        return sorted(default_ids), sorted(big_ids)
    elif mode == "missing":
        big_ids = []
        for qid in private_qs:
            resp = exp14_responses.get(qid)
            if resp is None or "\\boxed" not in (resp.get("response") or ""):
                big_ids.append(qid)
        return [], sorted(big_ids)
    else:
        raise ValueError(f"Unknown private mode: {mode}")


# ============================================================
# vLLM init + generation
# ============================================================
def init_vllm(cfg: dict):
    # Set env before importing vllm — caller (notebook or shell) should already
    # have done this for Kaggle, but be defensive locally too.
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    vllm_cfg = cfg["vllm"]
    model_id = cfg["model"]["model_id"]

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = "bfloat16" if bf16 else "float16"

    print(f"Loading tokenizer + vLLM engine: {model_id} (dtype={dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_id,
        dtype=dtype,
        tensor_parallel_size=vllm_cfg["tensor_parallel_size"],
        gpu_memory_utilization=vllm_cfg["gpu_memory_utilization"],
        max_model_len=vllm_cfg["max_model_len"],
        max_num_seqs=vllm_cfg.get("max_num_seqs", 32),
        max_num_batched_tokens=vllm_cfg.get("max_num_batched_tokens", 16384),
        trust_remote_code=True,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
    )
    return tokenizer, llm


def run_batch(llm, tokenizer, prompts_module, qs_subset: list[dict],
              n_samples: int, temperature: float, top_p: float, top_k: int,
              max_tokens: int):
    """Run N samples on each prompt. Returns list[list[str]] — one inner list
    of N response strings per input question."""
    from vllm import SamplingParams
    if not qs_subset:
        return []

    chat_prompts = [
        tokenizer.apply_chat_template(
            build_chat(q["question"], q.get("options"), prompts_module),
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in qs_subset
    ]

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    t0 = time.time()
    outputs = llm.generate(chat_prompts, sampling_params)
    elapsed = (time.time() - t0) / 60
    print(f"  generated {len(outputs)} x {n_samples} samples at "
          f"max_tokens={max_tokens} in {elapsed:.1f} min", flush=True)

    # outputs[i].outputs is a list of n_samples completions
    return [[c.text for c in o.outputs] for o in outputs]


# ============================================================
# Vote + merge
# ============================================================
def vote_one(samples: list[str], options: list | None, gold_count_hint: int,
              judger: Judger, min_cluster_size: int):
    """Vote on a list of sample responses. Returns:
       (winning_response_or_None, winning_cluster_size, cluster_sizes_list)"""
    boxed = [extract_boxed(s) for s in samples]
    clusters = cluster_by_judger(boxed, gold_count_hint, options, judger)
    cluster_sizes = sorted((len(c) for c in clusters), reverse=True)
    if not clusters:
        return None, 0, cluster_sizes
    best = max(clusters, key=len)
    if len(best) >= min_cluster_size:
        # Pick the first sample in the winning cluster as the canonical response
        winning_idx = best[0]
        return samples[winning_idx], len(best), cluster_sizes
    return None, len(best), cluster_sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["public", "private"], required=True)
    ap.add_argument("--private-mode", choices=["full", "missing"], default=None,
                    help="Override config.json selection.private_mode")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke-test: process at most N candidates per bucket")
    args = ap.parse_args()

    cfg = json.loads((EXP_DIR / "config.json").read_text())
    sampling = cfg["sampling"]
    voting = cfg["voting"]
    selection = cfg["selection"]
    n_samples = sampling["n_samples"]
    temp = sampling["temperature"]
    top_p = sampling["top_p"]
    top_k = sampling["top_k"]
    max_tokens_default = sampling["max_tokens_default"]
    max_tokens_big = sampling["max_tokens_missing_boxed"]
    min_cluster_size = voting["min_cluster_size"]

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif Path("/kaggle/working").exists():
        out_dir = Path("/kaggle/working")
    else:
        out_dir = EXP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}", flush=True)

    # Load prompts module
    prompts_path = EXP_DIR / "prompts.py"
    spec = importlib.util.spec_from_file_location("exp016_prompts", prompts_path)
    prompts_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts_module)

    # Load source exp_014 responses + competition data
    source_exp = cfg["source_experiment"]
    public_resp_path, private_resp_path = _find_stage1_responses(cfg)
    public_qs_path, private_qs_path = _find_competition_data()

    exp14_public = {r["id"]: r for r in (json.loads(l) for l in open(public_resp_path))}
    exp14_private = {r["id"]: r for r in (json.loads(l) for l in open(private_resp_path))}
    public_qs = {r["id"]: r for r in (json.loads(l) for l in open(public_qs_path))}
    private_qs = {r["id"]: r for r in (json.loads(l) for l in open(private_qs_path))}

    print(f"Loaded exp_14 responses: {len(exp14_public)} public / "
          f"{len(exp14_private)} private", flush=True)
    print(f"Competition data: {len(public_qs)} public / "
          f"{len(private_qs)} private", flush=True)

    judger = Judger()

    # ---- Candidate selection ----
    if args.target == "public":
        default_ids, big_ids = select_public_candidates(public_qs, exp14_public, judger)
        qs_lookup = public_qs
        exp14_lookup = exp14_public
    else:
        mode = args.private_mode or selection["private_mode"]
        default_ids, big_ids = select_private_candidates(private_qs, exp14_private, mode)
        qs_lookup = private_qs
        exp14_lookup = exp14_private

    if args.limit:
        default_ids = default_ids[: args.limit]
        big_ids = big_ids[: args.limit]
        print(f"LIMIT={args.limit} → smoke-testing on {len(default_ids)} + "
              f"{len(big_ids)} candidates", flush=True)

    print(f"\nCandidates: {len(default_ids)} default-budget + "
          f"{len(big_ids)} big-budget (missing_boxed)", flush=True)

    # ---- Skip vLLM entirely if no candidates ----
    if not default_ids and not big_ids:
        print("No candidates to process — emitting exp_14 outputs unchanged.", flush=True)
        _write_outputs(out_dir, args.target, exp14_public, exp14_private, {}, {})
        return

    tokenizer, llm = init_vllm(cfg)

    # ---- Generate ----
    def _generate_for(ids, max_tokens, label):
        if not ids:
            return {}
        qs_subset = [qs_lookup[i] for i in ids]
        print(f"\n[{label}] generating {len(ids)} x N={n_samples} at max_tokens={max_tokens}...",
              flush=True)
        samples_per_q = run_batch(
            llm, tokenizer, prompts_module, qs_subset,
            n_samples, temp, top_p, top_k, max_tokens,
        )
        return dict(zip(ids, samples_per_q))

    default_samples = _generate_for(default_ids, max_tokens_default, "default-budget")
    big_samples = _generate_for(big_ids, max_tokens_big, "big-budget")

    # ---- Vote + merge ----
    overrides_public: dict[int, str] = {}
    overrides_private: dict[int, str] = {}
    stats = {
        "target": args.target,
        "private_mode": (args.private_mode or selection["private_mode"]) if args.target == "private" else None,
        "n_default_candidates": len(default_ids),
        "n_big_candidates": len(big_ids),
        "flipped": 0,
        "no_majority_fallback": 0,
        "per_candidate": [],
    }

    for ids, samples_map, budget_label in [
        (default_ids, default_samples, "default"),
        (big_ids, big_samples, "big"),
    ]:
        for qid in ids:
            q = qs_lookup[qid]
            samples = samples_map[qid]
            gold = q.get("answer")
            gold_count = len(gold) if isinstance(gold, list) else 1
            winning_resp, winning_size, cluster_sizes = vote_one(
                samples, q.get("options"), gold_count, judger, min_cluster_size
            )
            entry = {
                "id": qid,
                "budget": budget_label,
                "cluster_sizes": cluster_sizes,
                "fell_back": winning_resp is None,
            }
            if winning_resp is None:
                stats["no_majority_fallback"] += 1
            else:
                stats["flipped"] += 1
                if args.target == "public":
                    overrides_public[qid] = winning_resp
                else:
                    overrides_private[qid] = winning_resp
            stats["per_candidate"].append(entry)

    print(f"\nVoting result: {stats['flipped']} overrides, "
          f"{stats['no_majority_fallback']} fall-backs to exp_14", flush=True)

    # ---- Write outputs ----
    _write_outputs(out_dir, args.target, exp14_public, exp14_private,
                   overrides_public, overrides_private)
    (out_dir / "best_of_n_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Wrote stats: {out_dir / 'best_of_n_stats.json'}", flush=True)


def _find_stage1_responses(cfg: dict):
    """Locate the upstream exp_14 responses (Kaggle dataset preferred, repo
    fallback for dev)."""
    import glob
    ds = cfg["stage1_dataset_name"]
    src = cfg["source_experiment"]

    def _look(filename):
        for p in glob.glob(f"/kaggle/input/{ds}/{filename}"):
            return Path(p)
        for p in glob.glob(f"/kaggle/input/**/{ds}/{filename}", recursive=True):
            return Path(p)
        # Local dev fallback — both regular file and gitignored .scored variants
        local = REPO / "experiments" / src / filename
        if local.exists():
            return local
        return None

    public = _look("public_responses.jsonl")
    private = _look("private_responses.jsonl")
    assert public, f"public_responses.jsonl not found for {src}"
    assert private, f"private_responses.jsonl not found for {src}"
    print(f"Stage-1 public:  {public}")
    print(f"Stage-1 private: {private}")
    return public, private


def _find_competition_data():
    """Locate public.jsonl + private.jsonl from Kaggle competitions path or
    repo data/ fallback."""
    kaggle = Path("/kaggle/input/competitions/cse-151-b-spring-2026-competition")
    if kaggle.exists():
        return kaggle / "public.jsonl", kaggle / "private.jsonl"
    return REPO / "data" / "public.jsonl", REPO / "data" / "private.jsonl"


def _write_outputs(out_dir: Path, target: str,
                    exp14_public: dict, exp14_private: dict,
                    overrides_public: dict, overrides_private: dict):
    """Write merged public_responses.jsonl + private_responses.jsonl, plus
    submission.csv from private."""

    def _merge(exp14_map: dict, overrides: dict) -> list[dict]:
        merged = []
        for qid in sorted(exp14_map.keys()):
            base = dict(exp14_map[qid])
            if qid in overrides:
                base["response"] = overrides[qid]
                base["_best_of_n_override"] = True
            else:
                base["_best_of_n_override"] = False
            merged.append(base)
        return merged

    public_merged = _merge(exp14_public, overrides_public)
    private_merged = _merge(exp14_private, overrides_private)

    public_out = out_dir / "public_responses.jsonl"
    private_out = out_dir / "private_responses.jsonl"
    sub_out = out_dir / "submission.csv"

    with open(public_out, "w") as f:
        for r in public_merged:
            f.write(json.dumps(r) + "\n")
    with open(private_out, "w") as f:
        for r in private_merged:
            f.write(json.dumps(r) + "\n")

    # submission.csv — id, response
    with open(sub_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "response"])
        for r in private_merged:
            w.writerow([r["id"], r["response"]])

    print(f"Wrote {public_out} ({len(public_merged)} rows)")
    print(f"Wrote {private_out} ({len(private_merged)} rows)")
    print(f"Wrote {sub_out}")


if __name__ == "__main__":
    main()
