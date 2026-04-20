"""Locally score a submission.csv or responses.jsonl against public.jsonl.

Useful when you have a Kaggle-produced submission CSV and want to know its
public-set accuracy without re-submitting.

Inputs:
    - submission.csv with columns (id, response), OR
    - responses.jsonl with fields (id, response) — is_mcq inferred from public
Output:
    - scored jsonl with (id, is_mcq, gold, response, correct)
    - accuracy printed to stdout

Usage:
    python scripts/score.py submission.csv --out experiments/exp_XXX/results.jsonl
    python scripts/score.py responses.jsonl
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text or "")
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", (text or "").upper())
    return matches[-1] if matches else ""


def load_responses(path: str) -> dict:
    p = Path(path)
    if p.suffix == ".csv":
        out = {}
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                out[int(row["id"])] = row.get("response", "")
        return out
    if p.suffix in (".jsonl", ".json"):
        out = {}
        for line in open(p):
            row = json.loads(line)
            out[row["id"]] = row.get("response", "")
        return out
    raise SystemExit(f"Unsupported file type: {p.suffix}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("responses", help="submission.csv or responses.jsonl")
    p.add_argument("--gold", default="data/public.jsonl")
    p.add_argument("--out", default=None,
                   help="Write scored jsonl here. Default: <responses stem>.scored.jsonl")
    args = p.parse_args()

    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    responses = load_responses(args.responses)
    gold_by_id = {json.loads(l)["id"]: json.loads(l) for l in open(args.gold)}

    shared = sorted(i for i in responses if i in gold_by_id)
    if not shared:
        raise SystemExit("No overlapping ids between responses and gold set.")
    print(f"Scoring {len(shared)} responses against {args.gold}...")

    out_path = args.out or str(Path(args.responses).with_suffix(".scored.jsonl"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    n_correct = n_mcq_c = n_mcq = n_free_c = n_free = 0
    with open(out_path, "w") as out:
        for qid in shared:
            q = gold_by_id[qid]
            resp = responses[qid]
            is_mcq = bool(q.get("options"))
            gold = q["answer"]

            if is_mcq:
                correct = extract_letter(resp) == str(gold).strip().upper()
                n_mcq += 1
                n_mcq_c += int(correct)
            else:
                gold_list = gold if isinstance(gold, list) else [gold]
                try:
                    correct = bool(judger.auto_judge(
                        pred=resp, gold=gold_list, options=[[]] * len(gold_list),
                    ))
                except Exception:
                    correct = False
                n_free += 1
                n_free_c += int(correct)

            n_correct += int(correct)
            out.write(json.dumps({
                "id": qid, "is_mcq": is_mcq, "gold": gold,
                "response": resp, "correct": correct,
            }) + "\n")

    def pct(a, b):
        return a / b * 100 if b else 0.0

    print(f"\n  Wrote → {out_path}")
    print("=" * 48)
    print(f"  Overall      {n_correct:4d}/{len(shared):4d}  {pct(n_correct, len(shared)):.2f}%")
    print(f"  MCQ          {n_mcq_c:4d}/{n_mcq:4d}  {pct(n_mcq_c, n_mcq):.2f}%")
    print(f"  Free-form    {n_free_c:4d}/{n_free:4d}  {pct(n_free_c, n_free):.2f}%")
    print("=" * 48)


if __name__ == "__main__":
    main()
