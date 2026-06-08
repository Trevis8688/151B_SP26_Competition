#!/usr/bin/env python3
"""exp_040 PROBE — PHASE 2 (CPU): sandbox + judge + report. NO GPU, no model.

Reads `probe_generations.jsonl` (written by probe_run.py), and for each completion:
  * PAL conditions: extract the final code block, run it through the sandbox
    (executor.py — already SIGKILL-timeout'd), assemble the tool answer into the box.
  * Judge the final response with a per-item SIGKILL timeout (safe_judge.judge_safe).

Writes `probe_outputs.jsonl` INCREMENTALLY (one line per item, flushed) so a crash
mid-loop preserves partial results — and `probe_report.json` + the PROBE REPORT to
stdout at the end.

WHY THIS IS SEPARATE FROM GENERATION (postcomp/DEVLOG.md 2026-06-07): the v1 probe
fused generate+judge; the judge hung in sympy for ~6h and the SIGKILL'd pod lost all
160 generations. Now generation is checkpointed first, and judging:
  (a) runs without the GPU — re-runnable on the login node or a Mac, no pod;
  (b) cannot hang — every judge is a fresh subprocess killed after `tool.judge_timeout_s`;
  (c) LOGS any id that trips the timeout — that id is the sympy-hang culprit we
      could not see when it took down the whole job.

GATE (notes.md): a PAL arm must emit >=60% runnable code AND net dev-FF accuracy
>= baseline (recovery >= regression). Fail -> redesign the prompt before scaling.

Usage:
  python probe_judge.py                     # uses sibling probe_generations.jsonl
  python probe_judge.py /path/to/gens.jsonl # explicit input
"""
import json
import os
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]
sys.path.insert(0, str(REPO))                            # judger
sys.path.insert(0, str(REPO / "postcomp" / "harness"))   # executor, pal, safe_judge

import pal                                                # noqa: E402
from safe_judge import judge_safe                         # noqa: E402

CFG = json.loads((EXP_DIR / "config.json").read_text())
GEN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP_DIR / "probe_generations.jsonl"
OUT_FILE = EXP_DIR / "probe_outputs.jsonl"
REPORT_FILE = EXP_DIR / "probe_report.json"

TOOL_TO = CFG["tool"]["timeout_s"]
TOOL_MEM = CFG["tool"]["mem_mb"]
JUDGE_TO = CFG["tool"].get("judge_timeout_s", 20.0)


def main():
    if not GEN_FILE.exists():
        sys.exit(f"[probe/judge] missing {GEN_FILE} — run probe_run.py (phase 1) first.")

    gens = [json.loads(l) for l in open(GEN_FILE) if l.strip()]
    conditions = (os.environ["PROBE_CONDITIONS"].split(",")
                  if os.environ.get("PROBE_CONDITIONS") else CFG["probe"]["conditions"])
    print(f"[probe/judge] {len(gens)} completions from {GEN_FILE.name}; "
          f"judge timeout={JUDGE_TO}s, tool timeout={TOOL_TO}s", flush=True)

    timed_out_ids = []  # the hang culprits we finally get to see
    rows = []
    # Stream outputs so a mid-loop crash still preserves what we judged.
    with open(OUT_FILE, "w") as fout:
        for i, g in enumerate(gens, 1):
            cond, qid, gold, text = g["condition"], g["id"], g["gold"], g["raw"]
            rec = {"condition": cond, "id": qid, "gold": gold,
                   "no_think_close": "</think>" not in text}  # degeneration proxy

            if cond in ("pal_nodemo", "pal_demo"):
                code = pal.last_code_block(text)
                rec["emitted_code"] = code is not None
                if code is not None:
                    ex = pal.run_code(code, timeout_s=TOOL_TO, mem_mb=TOOL_MEM)
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

            verdict = judge_safe(final, gold, timeout=JUDGE_TO)
            rec["correct"] = verdict["correct"]
            rec["judge_timed_out"] = verdict["timed_out"]
            rec["judge_poison"] = verdict.get("poison", False)
            if verdict["timed_out"]:
                timed_out_ids.append((cond, qid))
                print(f"[probe/judge] !! JUDGE TIMEOUT on cond={cond} id={qid} "
                      f"(>{JUDGE_TO}s) — counted wrong; this id is a sympy-hang culprit", flush=True)
            if verdict.get("poison"):
                print(f"[probe/judge] poison \\boxed{{}} (oversized) on cond={cond} id={qid} "
                      f"— rejected pre-judge, counted wrong", flush=True)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            rows.append(rec)
            if i % 20 == 0:
                print(f"[probe/judge] {i}/{len(gens)} judged", flush=True)

    # ── Summaries (same gate logic as the original probe) ──
    n = len({r["id"] for r in rows})
    base = {r["id"]: r["correct"] for r in rows if r["condition"] == "baseline"}
    report = {"n": n, "judge_timeouts": [{"condition": c, "id": i} for c, i in timed_out_ids],
              "conditions": {}}
    for cond in conditions:
        cr = [r for r in rows if r["condition"] == cond]
        if not cr:
            continue
        acc = sum(r["correct"] for r in cr)
        rep = {"accuracy": acc, "accuracy_pct": round(100 * acc / len(cr), 1),
               "no_think_close": sum(r.get("no_think_close", False) for r in cr),
               "no_think_close_pct": round(100 * sum(r.get("no_think_close", False) for r in cr) / len(cr), 1)}
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

    REPORT_FILE.write_text(json.dumps(report, indent=2))
    print("\n===== PROBE REPORT =====")
    print(json.dumps(report, indent=2))
    if timed_out_ids:
        print(f"\n[probe/judge] {len(timed_out_ids)} judge timeout(s): {timed_out_ids}")
    pal_pass = any(report["conditions"].get(c, {}).get("GATE_PASS") for c in ("pal_nodemo", "pal_demo"))
    print(f"\nGATE: {'PASS — proceed to full dev run' if pal_pass else 'FAIL — redesign prompt before scaling'}")


if __name__ == "__main__":
    main()
