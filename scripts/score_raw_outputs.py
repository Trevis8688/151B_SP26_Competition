#!/usr/bin/env python3
"""Score a raw_outputs JSONL file produced by sample_difficulty[_long].ipynb.

Input schema (per line):
  {"id": int, "is_mcq": bool, "answer": ..., "options": [...],
   "completion_texts": [str, ...], "completion_clipped": [bool, ...]}

Output schema (difficulty_samples format):
  {"id": int, "is_mcq": bool, "num_correct": int, "n_samples": int,
   "completions": [{"text": str, "correct": bool, "clipped": bool}, ...]}

A per-call timeout protects against pathological LaTeX that hangs Judger.auto_judge
(uses sympy under the hood — some inputs cause it to spin indefinitely).

Usage:
  python scripts/score_raw_outputs.py experiments/exp_009_grpo/raw_outputs_long.jsonl \
      --out experiments/exp_009_grpo/difficulty_samples_long.jsonl \
      --timeout 15
"""
import argparse
import json
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from judger import Judger


class JudgeTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise JudgeTimeout("auto_judge exceeded timeout")


def judge_with_timeout(judger, post, gold, opts_list, timeout_s):
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        return judger.auto_judge(post, gold, opts_list)
    finally:
        signal.alarm(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="path to raw_outputs JSONL")
    p.add_argument("--out", required=True, help="path to write difficulty_samples JSONL")
    p.add_argument("--timeout", type=int, default=15, help="seconds per judge call (default 15)")
    args = p.parse_args()

    judger = Judger()
    n_records = n_completions = n_timeout = n_correct_total = 0

    with open(args.input) as fin, open(args.out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            n_correct = 0
            comps = []
            for text, clipped in zip(r["completion_texts"], r["completion_clipped"]):
                idx = text.rfind("</think>")
                post = text[idx + len("</think>"):] if idx >= 0 else text
                gold = r["answer"]
                opts = r["options"]
                opts_list = ([opts] * len(gold)) if gold else [None]
                try:
                    ok = judge_with_timeout(judger, post, gold, opts_list, args.timeout)
                except JudgeTimeout:
                    ok = False
                    n_timeout += 1
                except Exception:
                    ok = False
                if ok:
                    n_correct += 1
                comps.append({"text": text, "correct": bool(ok), "clipped": bool(clipped)})
                n_completions += 1
            fout.write(json.dumps({
                "id": r["id"],
                "is_mcq": r["is_mcq"],
                "num_correct": n_correct,
                "n_samples": len(r["completion_texts"]),
                "completions": comps,
            }) + "\n")
            n_records += 1
            n_correct_total += n_correct
            if n_records % 50 == 0:
                print(f"  scored {n_records} records  "
                      f"({n_completions} completions, {n_timeout} timeouts, "
                      f"{n_correct_total}/{n_completions} correct)", flush=True)

    print(f"\nDone. {n_records} records, {n_completions} completions, "
          f"{n_timeout} timeouts ({100*n_timeout/n_completions:.1f}%), "
          f"{n_correct_total}/{n_completions} correct ({100*n_correct_total/n_completions:.1f}%)")


if __name__ == "__main__":
    main()
