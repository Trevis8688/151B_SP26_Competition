#!/usr/bin/env python3
"""Single entry point reproducing our competition submission end-to-end.

Pipeline (exp_018_pass2_rescue — Kaggle private LB 0.660 under the updated judge):

  Stage 1 — generation:
      Model  : TrevorDuong/qwen3-4b-thinking-grpo-pass2   (GRPO pass-2 policy)
      Prompt : SYSTEM + 3 MCQ few-shot examples (MCQ only) + question
      Sample : T=0.6, top_p=0.95, top_k=20, max_tokens=8192

  Stage 2 — rescue (only for responses that never emitted "\\boxed"):
      Model  : TrevorDuong/qwen3-4b-thinking-grpo-strict70  (GRPO pass-1 policy)
      Prompt : answer-extractor SYSTEM + (question + last 3000 tokens of the
               truncated stage-1 trace), no few-shots
      Sample : T=0.1, top_p=0.95, top_k=20, max_tokens=4096
      Merge  : append "[RESCUE EXTRACTION]" to the stage-1 text when the rescue
               run produced a "\\boxed".

  Output: submission.csv with columns (id, response) over all private IDs.

Calling `run_inference()` performs all of the above with no manual steps and
writes the final CSV. Both models load from the HuggingFace Hub.

PROCESS ISOLATION
-----------------
The pipeline loads two different models. Rather than tearing down vLLM's GPU
state in-process (notoriously version-fragile, and at gpu_memory_utilization=0.90
a single leaked allocation makes the second model OOM), `run_inference()` runs
each stage in a **fresh subprocess** of this same file. When a stage finishes,
the OS reclaims all of its VRAM on process exit — guaranteed, version-independent.
The orchestrator process never touches the GPU; it only spawns workers and
assembles the final CSV.

The prompts and sampling constants below are copied verbatim from
  experiments/exp_015_grpo_pass2/prompts.py   (stage-1, must match GRPO training)
  experiments/exp_012_boxed_rescue/prompts.py (rescue)
  experiments/exp_017_pass2_stage1/config.json + experiments/exp_018_pass2_rescue/config.json
and inlined here so this file is fully self-contained.
"""

import os

# ─── Must be set BEFORE importing vllm (applies in every worker subprocess) ───
# VLLM_USE_V1=0 selects the V0 engine, required on Kaggle T4 (SM 7.5) where the
# submission was produced, and harmless/ignored on newer builds. Because each
# stage runs in its own process, the pipeline does NOT rely on V0-specific
# teardown, so it still works if a newer V1-only vLLM ignores this flag.
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Models (already pushed to the HuggingFace Hub)
# ─────────────────────────────────────────────────────────────────────────────
STAGE1_MODEL = "TrevorDuong/qwen3-4b-thinking-grpo-pass2"
RESCUE_MODEL = "TrevorDuong/qwen3-4b-thinking-grpo-strict70"

# ─────────────────────────────────────────────────────────────────────────────
# Sampling / engine config (from exp_017 + exp_018 config.json)
# ─────────────────────────────────────────────────────────────────────────────
STAGE1_CFG = {
    "max_tokens": 8192,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_model_len": 10240,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 20480,
    "gpu_memory_utilization": 0.90,
}
RESCUE_CFG = {
    "max_tokens": 4096,
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 20,
    "max_model_len": 8192,
    "max_num_seqs": 24,
    "max_num_batched_tokens": 16384,
    "gpu_memory_utilization": 0.90,
    "max_input_tokens_from_stage1": 3000,
}

# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 prompts — copied from experiments/exp_015_grpo_pass2/prompts.py.
# The 3 MCQ few-shots MUST be present: the GRPO policy was trained with them, so
# removing them puts the model out-of-distribution and degrades MCQ accuracy.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)
FEWSHOT_MCQ = [
    {
        "role": "user",
        "content": (
            "Find 1 over 6 + 1 over 8.\n\n"
            "Options:\n"
            "A. 7 over 24\nB. 2 over 14\nC. 1 over 4\nD. 7 over 48\nE. 2 over 24\n"
            "F. 1 over 14\nG. 1 over 2\nH. 8 over 14\nI. 0.21 Repeating\nJ. 4 over 24"
        ),
    },
    {
        "role": "assistant",
        "content": "Common denominator is 24: 1/6 = 4/24 and 1/8 = 3/24, so 4/24 + 3/24 = 7/24. \\boxed{A}",
    },
    {
        "role": "user",
        "content": (
            "The function value of $\\cos(\\pi + 5i)$ is ( ).\n\n"
            "Options:\n"
            "A. -cosh5\nB. -sinh5\nC. sin5i\nD. -sin5\nE. cos5\n"
            "F. cosh5i\nG. sinh5\nH. -cos5\nI. cosh5\nJ. -sin5i"
        ),
    },
    {
        "role": "assistant",
        "content": "$\\cos(\\pi + 5i) = -\\cos(5i) = -\\cosh(5)$, using $\\cos(\\pi+x) = -\\cos x$ and $\\cos(ix) = \\cosh x$. \\boxed{A}",
    },
    {
        "role": "user",
        "content": (
            "Find the range of $f(x) = \\frac{ x }{ 1+x^2 }$.\n\n"
            "Options:\n"
            "A. -\\frac{1}{3} \\le y \\le \\frac{1}{3}\n"
            "B. -\\frac{1}{\\sqrt{3}} \\le y \\le \\frac{1}{\\sqrt{3}}\n"
            "C. -\\frac{1}{4} \\le y \\le \\frac{1}{4}\n"
            "D. -\\frac{1}{\\sqrt{2}} \\le y \\le \\frac{1}{\\sqrt{2}}\n"
            "E. -\\frac{1}{\\sqrt{6}} \\le y \\le \\frac{1}{\\sqrt{6}}\n"
            "F. -\\frac{1}{2} \\le y \\le \\frac{1}{2}\n"
            "G. -\\frac{1}{\\sqrt{5}} \\le y \\le \\frac{1}{\\sqrt{5}}\n"
            "H. -1 \\le y \\le 1\n"
            "I. -\\frac{1}{\\sqrt{7}} \\le y \\le \\frac{1}{\\sqrt{7}}\n"
            "J. -\\frac{1}{\\sqrt{4}} \\le y \\le \\frac{1}{\\sqrt{4}}"
        ),
    },
    {
        "role": "assistant",
        "content": "$f'(x) = \\frac{1-x^2}{(1+x^2)^2} = 0$ at $x=\\pm 1$, giving $f(\\pm 1) = \\pm 1/2$. So range is $-1/2 \\le y \\le 1/2$. \\boxed{F}",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Rescue prompts — copied from experiments/exp_012_boxed_rescue/prompts.py.
# ─────────────────────────────────────────────────────────────────────────────
RESCUE_SYSTEM_PROMPT_MATH = (
    "You are an answer extractor. The user provides a math problem and a partial "
    "reasoning trace that did not finish with a final answer. Based on the reasoning, "
    "output ONLY the final answer in \\boxed{}. If the problem has multiple sub-answers, "
    "separate them by commas inside a single \\boxed{}, e.g. \\boxed{3, 7}. "
    "Do not show your work. Do not restate the problem. Output only the box."
)
RESCUE_SYSTEM_PROMPT_MCQ = (
    "You are an answer extractor. The user provides a multiple-choice math problem "
    "and a partial reasoning trace that did not finish with a final answer. Based on "
    "the reasoning, output ONLY the chosen letter inside \\boxed{}, e.g. \\boxed{C}. "
    "Do not show your work. Do not restate the problem. Output only the box."
)


def build_rescue_user_message(question: str, options: Optional[list], partial_response: str) -> str:
    parts = [f"PROBLEM:\n{question}"]
    if options:
        parts.append("OPTIONS:\n" + "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)))
    parts.append(f"PARTIAL REASONING (incomplete):\n{partial_response}")
    parts.append(
        "Based on the reasoning above, output ONLY your final answer in \\boxed{}. "
        "Do not show your work."
    )
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Shared pure helpers (no GPU)
# ─────────────────────────────────────────────────────────────────────────────
def _needs_rescue(response: str) -> bool:
    return "\\boxed" not in (response or "")


def _load_jsonl(path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(rows: list, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _env_int(name: str, default: int) -> int:
    """Test-only override of a max_tokens constant via env var. Unset in normal
    runs → returns the submission default. Used by the smoke test to force short
    (no-\\boxed) stage-1 outputs so the stage-2 model load is exercised."""
    v = os.environ.get(name)
    return int(v) if v else default


def _default_tp() -> int:
    """Default to 1 GPU — the configuration the submission's stage-1 ran under,
    and the only value that is safe on every GPU. (tensor_parallel_size=2 has a
    documented CUDA-graph crash on Kaggle T4 / SM 7.5 that disable_custom_all_reduce
    does not fix; opt into multi-GPU explicitly via --tensor-parallel-size on
    Ampere+ hardware for speed.)"""
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# GPU workers — each runs in its own subprocess (heavy imports are local so the
# orchestrator process never initialises CUDA).
# ─────────────────────────────────────────────────────────────────────────────
def _load_llm(model_id: str, cfg: dict, tensor_parallel_size: int):
    from vllm import LLM
    return LLM(
        model=model_id,
        dtype="float16",  # matches the submission run (T4 has no bfloat16); fine on all GPUs
        tensor_parallel_size=tensor_parallel_size,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        max_model_len=cfg["max_model_len"],
        max_num_seqs=cfg["max_num_seqs"],
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        trust_remote_code=True,
    )


def _build_stage1_prompt(tokenizer, question: str, options: Optional[list]) -> str:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        system, user = SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
        fewshot = FEWSHOT_MCQ
    else:
        system, user = SYSTEM_PROMPT_MATH, question
        fewshot = []
    messages = [{"role": "system", "content": system}, *fewshot, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _worker_stage1(private_path: str, out_path: str, tp: int) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    data = _load_jsonl(private_path)
    print(f"[stage 1] {len(data)} questions; loading {STAGE1_MODEL} (tp={tp}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    llm = _load_llm(STAGE1_MODEL, STAGE1_CFG, tp)
    sampling = SamplingParams(
        n=1,
        max_tokens=_env_int("RUN_INFER_STAGE1_MAX_TOKENS", STAGE1_CFG["max_tokens"]),
        temperature=STAGE1_CFG["temperature"],
        top_p=STAGE1_CFG["top_p"],
        top_k=STAGE1_CFG["top_k"],
        min_p=0.0,
    )
    prompts = [_build_stage1_prompt(tokenizer, d["question"], d.get("options")) for d in data]
    print(f"[stage 1] generating {len(prompts)} responses ...", flush=True)
    outputs = llm.generate(prompts, sampling_params=sampling)
    responses = [
        {"id": d["id"], "is_mcq": bool(d.get("options")), "response": o.outputs[0].text.strip()}
        for d, o in zip(data, outputs)
    ]
    _write_jsonl(responses, out_path)
    n_missing = sum(_needs_rescue(r["response"]) for r in responses)
    print(f"[stage 1] done. {n_missing}/{len(responses)} missing \\boxed → rescue candidates. "
          f"Wrote {out_path}", flush=True)


def _worker_stage2(private_path: str, stage1_path: str, out_path: str, tp: int) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    responses = _load_jsonl(stage1_path)
    qs_by_id = {d["id"]: d for d in _load_jsonl(private_path)}
    candidates = [r for r in responses if _needs_rescue(r["response"])]

    if not candidates:
        print("[stage 2] no rescue candidates — copying stage-1 through.", flush=True)
        merged = [{**r, "rescued": False} for r in responses]
        _write_jsonl(merged, out_path)
        return

    print(f"[stage 2] {len(candidates)} candidates; loading {RESCUE_MODEL} (tp={tp}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(RESCUE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    llm = _load_llm(RESCUE_MODEL, RESCUE_CFG, tp)
    sampling = SamplingParams(
        n=1,
        max_tokens=_env_int("RUN_INFER_STAGE2_MAX_TOKENS", RESCUE_CFG["max_tokens"]),
        temperature=RESCUE_CFG["temperature"],
        top_p=RESCUE_CFG["top_p"],
        top_k=RESCUE_CFG["top_k"],
        min_p=0.0,
    )

    n_in = RESCUE_CFG["max_input_tokens_from_stage1"]

    def truncate(text: str) -> str:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= n_in:
            return text
        return tokenizer.decode(ids[-n_in:], skip_special_tokens=True)

    prompts = []
    for r in candidates:
        q = qs_by_id[r["id"]]
        system = RESCUE_SYSTEM_PROMPT_MCQ if r["is_mcq"] else RESCUE_SYSTEM_PROMPT_MATH
        user = build_rescue_user_message(q["question"], q.get("options"), truncate(r["response"]))
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
        )

    print(f"[stage 2] generating {len(prompts)} rescue extractions ...", flush=True)
    outputs = llm.generate(prompts, sampling_params=sampling)

    rescue_by_id = {}
    for r, o in zip(candidates, outputs):
        text = o.outputs[0].text.strip()
        if "\\boxed" in text:
            rescue_by_id[r["id"]] = text

    merged = []
    for r in responses:
        new_r = dict(r)
        if r["id"] in rescue_by_id:
            new_r["response"] = r["response"] + "\n\n[RESCUE EXTRACTION]:\n" + rescue_by_id[r["id"]]
            new_r["rescued"] = True
        else:
            new_r["rescued"] = False
        merged.append(new_r)
    _write_jsonl(merged, out_path)
    print(f"[stage 2] rescued {len(rescue_by_id)}/{len(candidates)} candidates. Wrote {out_path}",
          flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator (no GPU) + CSV writer
# ─────────────────────────────────────────────────────────────────────────────
def _write_submission(responses: list, out_csv: str) -> None:
    by_id = {r["id"]: r["response"] for r in responses}
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "response"])
        for qid in sorted(by_id.keys()):
            writer.writerow([qid, by_id[qid]])
    print(f"\nWrote submission: {out_path}  ({len(by_id)} rows)", flush=True)


def _spawn_worker(stage: str, tp: int, **paths) -> None:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", stage,
           "--tensor-parallel-size", str(tp)]
    for k, v in paths.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    print(f"\n=== spawning {stage} subprocess ===\n{' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def run_inference(
    private_path: str = "data/private.jsonl",
    output_csv: str = "submission.csv",
    tensor_parallel_size: Optional[int] = None,
    work_dir: Optional[str] = None,
) -> str:
    """Run the full two-stage pipeline on the private set and write submission.csv.

    Each stage runs in a fresh subprocess so GPU memory is fully reclaimed
    between the two model loads (see PROCESS ISOLATION in the module docstring).

    Args:
        private_path: path to private.jsonl (the leaderboard test set, no answers).
        output_csv:   path to write the final submission CSV.
        tensor_parallel_size: GPUs for vLLM. Defaults to 1 (safe on all GPUs;
            pass 2 on Ampere+ multi-GPU for speed — see _default_tp).
        work_dir:     directory for intermediate JSONL (default: alongside output_csv).

    Returns:
        The path to the written submission CSV.
    """
    if tensor_parallel_size is None:
        tensor_parallel_size = _default_tp()

    work = Path(work_dir) if work_dir else (Path(output_csv).resolve().parent / ".run_inference_work")
    work.mkdir(parents=True, exist_ok=True)
    stage1_path = work / "stage1_responses.jsonl"
    merged_path = work / "merged_responses.jsonl"

    private_path = str(Path(private_path).resolve())
    print(f"Pipeline start | private={private_path} | tp={tensor_parallel_size} | work={work}",
          flush=True)

    _spawn_worker("stage1", tensor_parallel_size,
                  private_path=private_path, out_path=str(stage1_path))
    _spawn_worker("stage2", tensor_parallel_size,
                  private_path=private_path, stage1_path=str(stage1_path),
                  out_path=str(merged_path))

    merged = _load_jsonl(merged_path)
    _write_submission(merged, output_csv)
    return output_csv


# ─────────────────────────────────────────────────────────────────────────────
# CLI — dispatches to the orchestrator or to a single GPU worker.
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Reproduce the competition submission end-to-end.")
    ap.add_argument("--private-path", default="data/private.jsonl",
                    help="Path to private.jsonl (default: data/private.jsonl)")
    ap.add_argument("--output-csv", default="submission.csv",
                    help="Where to write the submission CSV (default: submission.csv)")
    ap.add_argument("--tensor-parallel-size", type=int, default=None,
                    help="GPUs for vLLM (default: 1; pass 2 on Ampere+ multi-GPU for speed)")
    ap.add_argument("--work-dir", default=None,
                    help="Directory for intermediate JSONL (default: next to --output-csv)")
    # Internal: select a single-stage GPU worker (used by the orchestrator).
    ap.add_argument("--worker", choices=["stage1", "stage2"], default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--out-path", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--stage1-path", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker == "stage1":
        tp = args.tensor_parallel_size or _default_tp()
        _worker_stage1(args.private_path, args.out_path, tp)
    elif args.worker == "stage2":
        tp = args.tensor_parallel_size or _default_tp()
        _worker_stage2(args.private_path, args.stage1_path, args.out_path, tp)
    else:
        run_inference(
            private_path=args.private_path,
            output_csv=args.output_csv,
            tensor_parallel_size=args.tensor_parallel_size,
            work_dir=args.work_dir,
        )


if __name__ == "__main__":
    main()
