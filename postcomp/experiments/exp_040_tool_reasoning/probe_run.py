#!/usr/bin/env python3
"""exp_040 go/no-go PROBE — 40-question dev free-form mini-eval (DSMLP, vLLM).

One run answers all the Phase-1 open questions:
  1. Does Qwen3-4B-Thinking emit a *runnable* final code block when asked? (per arm)
  2. Which prompt variant wins: baseline / fullprec_A / pal_nodemo / pal_demo?
  3. Does the full PAL pipeline net-improve dev FF accuracy (recovery vs regression)?

For each of 40 dev free-form questions (9 known precision-recoverable forced in +
random fill, seed 42) it generates under every condition, runs PAL conditions through
the real sandbox (executor.py → pal.py), and judges every result with the repo judger.

GATE (notes.md): a PAL arm must emit >=60% runnable code AND net dev-FF accuracy
>= baseline (recovery >= regression). If it fails, redesign the prompt before scaling.

Run on DSMLP A5000 via scripts/launch_exp040_probe.sh. Writes:
  probe_outputs.jsonl  — every generation + per-item eval
  probe_report.json    — per-condition summary + the gate verdict
"""
import os

# Must be set BEFORE torch/vllm import (vllm is imported inside main()). Reduces
# allocator fragmentation — the bug-040 OOM left 1.22 GiB reserved-but-unallocated.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import random
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]
sys.path.insert(0, str(REPO))                       # judger
sys.path.insert(0, str(REPO / "postcomp" / "harness"))  # executor, pal

import prompts as P                                  # noqa: E402  (exp-local prompts.py)
from judger import Judger                            # noqa: E402
import pal                                           # noqa: E402

CFG = json.loads((EXP_DIR / "config.json").read_text())
J = Judger(strict_extract=False)


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


def judge(response, gold):
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return bool(J.auto_judge(pred=response, gold=gold_list, options=[[]] * len(gold_list)))
    except Exception:
        return False


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sample = load_dev_ff_sample()
    conditions = CFG["probe"]["conditions"]
    print(f"[probe] {len(sample)} dev FF questions x {len(conditions)} conditions "
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
    llm = LLM(
        model=CFG["model_id"],
        dtype=v.get("dtype", "bfloat16"),
        gpu_memory_utilization=v["gpu_memory_utilization"],
        max_model_len=v["max_model_len"],
        max_num_seqs=v["max_num_seqs"],
        max_num_batched_tokens=v["max_num_batched_tokens"],
        enforce_eager=v.get("enforce_eager", False),  # bug-040: frees CUDA-graph pools on A5000
        trust_remote_code=True,
    )
    sp = SamplingParams(n=1, max_tokens=CFG["max_tokens"], temperature=CFG["temperature"],
                        top_p=CFG["top_p"], top_k=CFG["top_k"], min_p=0.0)

    print("[probe] generating ...", flush=True)
    outs = llm.generate([t[3] for t in tagged], sampling_params=sp)

    tool_to = CFG["tool"]["timeout_s"]
    tool_mem = CFG["tool"]["mem_mb"]

    rows = []
    for (cond, qid, gold, _), o in zip(tagged, outs):
        text = o.outputs[0].text.strip()
        rec = {"condition": cond, "id": qid, "gold": gold, "raw": text}
        if cond in ("pal_nodemo", "pal_demo"):
            code = pal.last_code_block(text)
            rec["emitted_code"] = code is not None
            if code is not None:
                ex = pal.run_code(code, timeout_s=tool_to, mem_mb=tool_mem)
                outcome = pal.assemble(text, ex)
                rec["runnable"] = ex.ok
                rec["answer_line"] = pal.parse_tool_answer(ex.stdout) is not None if ex.ok else False
                rec["used_tool"] = outcome.used_tool
                rec["code_after_think"] = ("</think>" in text) and (text.rfind("```") > text.rfind("</think>"))
                final = outcome.response
            else:
                rec["runnable"] = rec["answer_line"] = rec["used_tool"] = rec["code_after_think"] = False
                final = text
        else:
            final = text
        rec["correct"] = judge(final, gold)
        rows.append(rec)

    (EXP_DIR / "probe_outputs.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    # ── Summaries ──
    base = {r["id"]: r["correct"] for r in rows if r["condition"] == "baseline"}
    report = {"n": len(sample), "conditions": {}}
    for cond in conditions:
        cr = [r for r in rows if r["condition"] == cond]
        acc = sum(r["correct"] for r in cr)
        rep = {"accuracy": acc, "accuracy_pct": round(100 * acc / len(cr), 1)}
        if cond != "baseline":
            rep["recovery"] = sorted(r["id"] for r in cr if r["correct"] and not base.get(r["id"], False))
            rep["regression"] = sorted(r["id"] for r in cr if not r["correct"] and base.get(r["id"], False))
            rep["net_vs_baseline"] = len(rep["recovery"]) - len(rep["regression"])
        if cond in ("pal_nodemo", "pal_demo"):
            rep["emitted_code_pct"] = round(100 * sum(r["emitted_code"] for r in cr) / len(cr), 1)
            rep["runnable_pct"] = round(100 * sum(r.get("runnable", False) for r in cr) / len(cr), 1)
            rep["used_tool_pct"] = round(100 * sum(r.get("used_tool", False) for r in cr) / len(cr), 1)
            rep["code_after_think_pct"] = round(100 * sum(r.get("code_after_think", False) for r in cr) / len(cr), 1)
            rep["gate_runnable_ok"] = rep["runnable_pct"] >= 60.0
            rep["gate_net_ok"] = rep["net_vs_baseline"] >= 0
            rep["GATE_PASS"] = rep["gate_runnable_ok"] and rep["gate_net_ok"]
        report["conditions"][cond] = rep

    (EXP_DIR / "probe_report.json").write_text(json.dumps(report, indent=2))
    print("\n===== PROBE REPORT =====")
    print(json.dumps(report, indent=2))
    pal_pass = any(report["conditions"].get(c, {}).get("GATE_PASS") for c in ("pal_nodemo", "pal_demo"))
    print(f"\nGATE: {'PASS — proceed to full dev run' if pal_pass else 'FAIL — redesign prompt before scaling'}")


if __name__ == "__main__":
    main()
