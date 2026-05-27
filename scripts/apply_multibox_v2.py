"""exp_037: extended multi-box judger-extraction fix (rule_v4).

Strict superset of exp_035 (apply_multibox_fix.py). exp_035 only handled
under-extraction; rule_v4 also handles cases where the model wrote a
summary-style trailing box that the judger greedily groups with preceding
boxes, producing over-extraction.

Rule_v4:
  n_exp = number of `[ANS]` markers in the question (predicts gold length
          with 97.6% accuracy on public).
  If n_exp < 2, leave alone.
  Let boxes = all `\\boxed{}` contents in order. If empty, leave alone.
  Let cur_n = number of items the judger currently extracts.
  If cur_n == n_exp, leave alone (already matching, don't clobber).

  Else try in order:
    A) Last single box's comma-split count == n_exp -> append
       "Final Answer: \\boxed{<last_box>}" so the judger picks ONLY that box.
    B) len(boxes) >= n_exp -> append "Final Answer: \\boxed{...} \\boxed{...}"
       with the last n_exp boxes (exp_035's rule, no internal-comma constraint).
    Else leave alone.

On exp_018 public (1126 q, FF only): rec=13, brk=0, ties=48 (vs exp_035 rec=9).
Net +4 recoveries, distribution-free, additive-only. Subsumes exp_035 entirely
(candidate B == exp_035's rule; candidate A only adds previously-skipped cases).

Usage:
    python scripts/apply_multibox_v2.py \
        --responses experiments/exp_018_pass2_rescue/private_responses.jsonl \
        --questions data/private.jsonl \
        --out_responses experiments/exp_037_multibox_v2/private_responses.jsonl \
        --out_submission experiments/exp_037_multibox_v2/submission.csv
"""
import argparse
import csv
import json
import re
import signal
import sys
from pathlib import Path

BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
ANS_RE = re.compile(r"\[ANS\]", re.IGNORECASE)


def _timeout(s, f):
    raise TimeoutError()


def current_extract_count(judger, resp: str) -> int:
    if not resp:
        return 0
    try:
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(5)
        extracted = judger.extract_ans(resp)
        signal.alarm(0)
    except Exception:
        signal.alarm(0)
        return 0
    if not extracted:
        return 0
    return len([p for p in judger.split_by_comma(extracted) if p.strip()])


def fix_one(judger, resp: str, question_text: str) -> tuple[str, bool, str]:
    """Apply rule_v3. Return (new_resp, modified, which_candidate)."""
    n_exp = len(ANS_RE.findall(question_text or ""))
    if n_exp < 2:
        return resp, False, "skip_n_ans_lt_2"
    boxes = BOX_RE.findall(resp or "")
    if not boxes:
        return resp, False, "skip_no_boxes"
    cur_n = current_extract_count(judger, resp)
    if cur_n == n_exp:
        return resp, False, "skip_already_matches"

    last = boxes[-1]
    last_parts = [p.strip() for p in judger.split_by_comma(last) if p.strip()]
    if len(last_parts) == n_exp:
        new_resp = (resp or "").rstrip() + f"\n\nFinal Answer: \\boxed{{{last}}}"
        return new_resp, True, "A_last_box"

    if len(boxes) >= n_exp:
        last_n = boxes[-n_exp:]
        new_resp = (resp or "").rstrip() + "\n\nFinal Answer: " + " ".join(
            f"\\boxed{{{b}}}" for b in last_n
        )
        return new_resp, True, "B_last_n"

    return resp, False, "skip_no_candidate"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out_responses", required=True)
    ap.add_argument("--out_submission", required=True)
    ap.add_argument("--judger_dir", default=".")
    args = ap.parse_args()

    if args.judger_dir not in sys.path:
        sys.path.insert(0, args.judger_dir)
    from judger import Judger
    judger = Judger(strict_extract=False)

    questions = {q["id"]: q for q in (json.loads(l) for l in open(args.questions))}
    rows = [json.loads(l) for l in open(args.responses)]
    print(f"Loaded {len(rows)} responses; {len(questions)} questions.")

    out_rows = []
    candidate_counts: dict[str, int] = {}
    for r in rows:
        qid = r["id"]
        q = questions.get(qid)
        if q is None:
            out_rows.append(r); continue
        if q.get("options"):
            candidate_counts["skip_mcq"] = candidate_counts.get("skip_mcq", 0) + 1
            out_rows.append(r); continue
        new_resp, modified, label = fix_one(judger, r["response"], q["question"])
        candidate_counts[label] = candidate_counts.get(label, 0) + 1
        if modified:
            r = {**r, "response": new_resp}
        out_rows.append(r)

    n_modified = candidate_counts.get("A_last_box", 0) + candidate_counts.get("B_last_n", 0)
    print(f"\nModified: {n_modified} responses ({n_modified/len(rows)*100:.2f}% of total).")
    print(f"Candidate breakdown:")
    for k, v in sorted(candidate_counts.items()):
        print(f"  {k:24s}  {v:4d}")

    out_responses = Path(args.out_responses)
    out_responses.parent.mkdir(parents=True, exist_ok=True)
    with open(out_responses, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote responses: {out_responses}")

    out_csv = Path(args.out_submission)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r["response"] for r in out_rows}
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "response"])
        for qid in sorted(by_id.keys()):
            w.writerow([qid, by_id[qid]])
    print(f"Wrote submission: {out_csv}  ({len(by_id)} rows)")


if __name__ == "__main__":
    main()
