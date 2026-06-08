#!/usr/bin/env python3
"""exp_040 PROBE — PHASE 1 (GPU): generate + checkpoint. NO judging here.

Generates 40 dev free-form questions x 4 conditions (baseline / fullprec_A /
pal_nodemo / pal_demo) = 160 completions on DSMLP A5000, then writes every raw
completion to `probe_generations.jsonl` and EXITS. Judging/sandbox happens in
phase 2 (probe_judge.py), which is CPU-only and runs without the GPU.

WHY SPLIT (postcomp/DEVLOG.md 2026-06-07): the first probe generated all 160
completions, then the judging loop hung inside sympy for ~6h until the pod's
wall-clock kill — and because outputs were only written *after* judging, all 160
generations (32 min of GPU) were lost. Lesson: checkpoint the expensive GPU work
to disk the instant it exists; never let a downstream CPU step put it at risk.

Phase 2 (`probe_judge.py`) reads `probe_generations.jsonl` from the shared PVC and
judges with a per-item SIGKILL timeout (safe_judge.py) — so a sympy hang can never
again destroy a generation or stall for 6h.

Run on DSMLP A5000 via scripts/launch_exp040_probe.sh (which runs phase 2 after).
Writes: probe_generations.jsonl  — {condition, id, gold, raw} per completion.
"""
import os

# Must be set BEFORE torch/vllm import. Reduces allocator fragmentation
# (the bug-040 OOM left 1.22 GiB reserved-but-unallocated).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Engine selector for the known-good-match discriminator. run_inference.py (0.581/0.660)
# runs the V0 engine (VLLM_USE_V1=0); the probe defaulted to V1 (vLLM 0.8.5 default).
# dtype was already ruled out (fp16 degenerates identically), so PROBE_V0=1 tests the
# last remaining divergence — whether the V1 sampling path (PyTorch-native top-p/top-k
# fallback) is what collapses long generations into "0 0 0". Must precede the vllm import.
if os.environ.get("PROBE_V0") == "1":
    os.environ["VLLM_USE_V1"] = "0"

import json
import random
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]
sys.path.insert(0, str(REPO))                            # (repo root, for parity)
sys.path.insert(0, str(REPO / "postcomp" / "harness"))   # prompts live next to exp

import prompts as P                                       # noqa: E402  (exp-local prompts.py)

CFG = json.loads((EXP_DIR / "config.json").read_text())
GEN_FILE = EXP_DIR / "probe_generations.jsonl"


def load_dev_ff_sample():
    dev = [json.loads(l) for l in open(REPO / "data" / "splits" / "dev.jsonl")]
    ff = [d for d in dev if not d.get("options")]
    by_id = {d["id"]: d for d in ff}
    forced = [i for i in CFG["probe"]["force_include_recoverable"] if i in by_id]
    rest = [d["id"] for d in ff if d["id"] not in forced]
    random.Random(CFG["probe"]["sample_seed"]).shuffle(rest)
    n = CFG["probe"]["sample_size"]
    chosen = forced + rest[: max(0, n - len(forced))]
    return [by_id[i] for i in chosen]


def build_messages(condition, question):
    """System + optional demo + user message for a free-form question."""
    if condition == "baseline":
        system, demo = P.SYSTEM_PROMPT_MATH, []
    elif condition == "fullprec_A":
        system, demo = P.SYSTEM_PROMPT_MATH_FULLPREC, []
    elif condition == "pal_nodemo":
        system, demo = P.SYSTEM_PROMPT_MATH_PAL, []
    elif condition == "pal_demo":
        system, demo = P.SYSTEM_PROMPT_MATH_PAL, P.FEWSHOT_PAL_DEMO
    else:
        raise ValueError(condition)
    return [{"role": "system", "content": system}, *demo, {"role": "user", "content": question}]


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sample = load_dev_ff_sample()
    # Env overrides let a quick discriminator run a subset / different dtype without
    # editing committed config (e.g. PROBE_CONDITIONS=baseline PROBE_DTYPE=float16).
    conditions = (os.environ["PROBE_CONDITIONS"].split(",")
                  if os.environ.get("PROBE_CONDITIONS") else CFG["probe"]["conditions"])
    print(f"[probe/gen] {len(sample)} dev FF questions x {len(conditions)} conditions "
          f"= {len(sample)*len(conditions)} generations", flush=True)

    tok = AutoTokenizer.from_pretrained(CFG["model_id"])
    tok.pad_token = tok.eos_token

    # Build all prompts, tagged so we can route outputs back.
    tagged = []  # (cond, id, gold, prompt_str)
    for cond in conditions:
        for q in sample:
            msgs = build_messages(cond, q["question"])
            ps = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            tagged.append((cond, q["id"], q["answer"], ps))

    v = CFG["vllm"]
    dtype = os.environ.get("PROBE_DTYPE") or v.get("dtype", "bfloat16")
    print(f"[probe/gen] dtype={dtype} conditions={conditions}", flush=True)
    llm = LLM(
        model=CFG["model_id"],
        dtype=dtype,
        gpu_memory_utilization=v["gpu_memory_utilization"],
        max_model_len=v["max_model_len"],
        max_num_seqs=v["max_num_seqs"],
        max_num_batched_tokens=v["max_num_batched_tokens"],
        enforce_eager=v.get("enforce_eager", False),  # bug-040: frees CUDA-graph pools on A5000
        trust_remote_code=True,
    )
    sp = SamplingParams(n=1, max_tokens=CFG["max_tokens"], temperature=CFG["temperature"],
                        top_p=CFG["top_p"], top_k=CFG["top_k"], min_p=0.0)

    print("[probe/gen] generating ...", flush=True)
    outs = llm.generate([t[3] for t in tagged], sampling_params=sp)

    # ── CHECKPOINT IMMEDIATELY — the expensive GPU work is now safe on disk. ──
    rows = []
    for (cond, qid, gold, _), o in zip(tagged, outs):
        rows.append({"condition": cond, "id": qid, "gold": gold,
                     "raw": o.outputs[0].text.strip()})
    GEN_FILE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    # Degeneration check (bug under diagnosis 2026-06-07): completions that never close
    # </think> ran to the token budget and collapsed into repetition. ~50% on bf16/V1;
    # this line is the discriminator readout when re-run under PROBE_DTYPE=float16.
    no_close = sum(1 for r in rows if "</think>" not in r["raw"])
    print(f"[probe/gen] degeneration check: {no_close}/{len(rows)} never closed </think> "
          f"(dtype={dtype})", flush=True)
    print(f"[probe/gen] wrote {len(rows)} completions -> {GEN_FILE.name}", flush=True)
    print("[probe/gen] phase 1 done. Run probe_judge.py next "
          "(CPU-only; safe to run on the login node).", flush=True)


if __name__ == "__main__":
    main()
