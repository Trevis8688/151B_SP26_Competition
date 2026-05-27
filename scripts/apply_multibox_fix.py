"""Apply the multi-box extraction fix to a *_responses.jsonl + question file.

Why: the judger's `extract_all_boxed` only takes the LAST CONTIGUOUS GROUP of
\\boxed{} expressions. Multi-part responses with boxes separated by prose
(e.g. "Part i): $\\boxed{A}$ - Part ii): $\\boxed{C}$") collapse to a single
trailing box, so multi-gold scoring loses to a length mismatch.

Strict rule (verified on exp_018 public: 0 breaks / 8 recoveries / +0.71pp):
  1. n_expected = count of `[ANS]` markers in the question. (Public matches
     gold length 733/751 = 97.6%; the few mismatches are 1-ANS questions with
     a list-style gold, which the rule simply leaves alone.)
  2. If n_expected < 2, leave the response alone.
  3. Collect all \\boxed{} expressions from the response, in order.
  4. If fewer than n_expected boxes exist, leave the response alone (not
     enough material to construct a multi-part answer).
  5. Compute what the judger CURRENTLY extracts from the response.
  6. If the current extraction already has >= n_expected items, leave alone.
     (The judger is already finding the answers; don't clobber what works.)
  7. Otherwise, append "\\n\\nFinal Answer: \\boxed{...} \\boxed{...} ..." with
     the LAST n_expected boxes — a contiguous trailing group the judger picks
     up cleanly.

Usage:
    python scripts/apply_multibox_fix.py \
        --responses experiments/exp_018_pass2_rescue/private_responses.jsonl \
        --questions data/private.jsonl \
        --out_responses experiments/exp_035_multibox_fix/private_responses.jsonl \
        --out_submission experiments/exp_035_multibox_fix/submission.csv
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
ANS_RE = re.compile(r"\[ANS\]", re.IGNORECASE)


def current_extract_count(judger, resp: str) -> int:
    """How many items would the judger currently extract from this response?"""
    if not resp:
        return 0
    extracted = judger.extract_ans(resp)
    if not extracted:
        return 0
    return len([x for x in extracted.split(",") if x.strip()])


def fix_one(judger, resp: str, question_text: str) -> tuple[str, bool]:
    """Apply the strict rule. Return (new_resp, modified_bool)."""
    n_expected = len(ANS_RE.findall(question_text or ""))
    if n_expected < 2:
        return resp, False
    boxes = BOX_RE.findall(resp or "")
    if len(boxes) < n_expected:
        return resp, False
    cur_n = current_extract_count(judger, resp)
    if cur_n >= n_expected:
        return resp, False
    last_n = boxes[-n_expected:]
    new_resp = (resp or "").rstrip() + "\n\nFinal Answer: " + " ".join(
        f"\\boxed{{{b}}}" for b in last_n
    )
    return new_resp, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="input *_responses.jsonl")
    ap.add_argument("--questions", required=True, help="data/public.jsonl or data/private.jsonl")
    ap.add_argument("--out_responses", required=True, help="output *_responses.jsonl with fix applied")
    ap.add_argument("--out_submission", required=True, help="output submission.csv")
    ap.add_argument("--judger_dir", default=".", help="dir containing judger.py")
    args = ap.parse_args()

    if args.judger_dir not in sys.path:
        sys.path.insert(0, args.judger_dir)
    from judger import Judger
    judger = Judger(strict_extract=False)

    questions = {q["id"]: q for q in (json.loads(l) for l in open(args.questions))}
    rows = [json.loads(l) for l in open(args.responses)]
    print(f"Loaded {len(rows)} responses; {len(questions)} questions.")

    out_rows = []
    n_modified = 0
    n_mcq_skipped = 0
    for r in rows:
        qid = r["id"]
        q = questions.get(qid)
        if q is None:
            out_rows.append(r)
            continue
        if q.get("options"):
            n_mcq_skipped += 1
            out_rows.append(r)
            continue
        new_resp, modified = fix_one(judger, r["response"], q["question"])
        if modified:
            n_modified += 1
            r = {**r, "response": new_resp}
        out_rows.append(r)

    print(f"Modified: {n_modified} responses ({n_modified/len(rows)*100:.2f}% of total).")
    print(f"MCQ skipped (rule does not apply): {n_mcq_skipped}.")

    out_responses = Path(args.out_responses)
    out_responses.parent.mkdir(parents=True, exist_ok=True)
    with open(out_responses, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote responses with fix: {out_responses}")

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
