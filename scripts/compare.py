"""Diff two results.jsonl files: which questions flipped correct↔wrong.

Expects both files to cover the same set of ids (or an overlapping subset —
only shared ids are compared).

Usage:
    python scripts/compare.py baseline.jsonl new.jsonl
    python scripts/compare.py a.jsonl b.jsonl --show 10
    python scripts/compare.py a.jsonl b.jsonl --json
"""
import argparse
import json
import textwrap


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("new")
    p.add_argument("--show", type=int, default=5,
                   help="How many gained/regressed questions to preview.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    a = load(args.baseline)
    b = load(args.new)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("No shared ids between the two files.")

    gains = [i for i in shared if not a[i]["correct"] and b[i]["correct"]]
    regressions = [i for i in shared if a[i]["correct"] and not b[i]["correct"]]
    unchanged_correct = [i for i in shared if a[i]["correct"] and b[i]["correct"]]
    unchanged_wrong = [i for i in shared if not a[i]["correct"] and not b[i]["correct"]]

    a_acc = sum(a[i]["correct"] for i in shared) / len(shared) * 100
    b_acc = sum(b[i]["correct"] for i in shared) / len(shared) * 100

    mcq = [i for i in shared if a[i].get("is_mcq")]
    free = [i for i in shared if not a[i].get("is_mcq")]
    def split_acc(ids, src):
        return sum(src[i]["correct"] for i in ids) / len(ids) * 100 if ids else 0.0

    summary = {
        "shared_questions": len(shared),
        "baseline_acc": a_acc,
        "new_acc": b_acc,
        "delta": b_acc - a_acc,
        "mcq": {
            "baseline": split_acc(mcq, a),
            "new": split_acc(mcq, b),
            "delta": split_acc(mcq, b) - split_acc(mcq, a),
        },
        "free": {
            "baseline": split_acc(free, a),
            "new": split_acc(free, b),
            "delta": split_acc(free, b) - split_acc(free, a),
        },
        "gains": len(gains),
        "regressions": len(regressions),
        "unchanged_correct": len(unchanged_correct),
        "unchanged_wrong": len(unchanged_wrong),
        "gain_ids": gains,
        "regression_ids": regressions,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("=" * 56)
    print(f"  {len(shared)} shared questions")
    print(f"  baseline:  {a_acc:5.2f}%      new:  {b_acc:5.2f}%      Δ {b_acc-a_acc:+.2f}%")
    print(f"    MCQ:     {summary['mcq']['baseline']:5.2f}% → {summary['mcq']['new']:5.2f}%   Δ {summary['mcq']['delta']:+.2f}%")
    print(f"    Free:    {summary['free']['baseline']:5.2f}% → {summary['free']['new']:5.2f}%   Δ {summary['free']['delta']:+.2f}%")
    print("=" * 56)
    print(f"  Gains       (wrong → correct): {len(gains)}")
    print(f"  Regressions (correct → wrong): {len(regressions)}")
    print(f"  Stayed correct: {len(unchanged_correct)}    Stayed wrong: {len(unchanged_wrong)}")
    print("=" * 56)

    def preview(ids, label):
        if not ids:
            return
        print(f"\n── First {min(args.show, len(ids))} {label} ──")
        for i in ids[: args.show]:
            g = a[i].get("gold")
            tag = "MCQ" if a[i].get("is_mcq") else "FREE"
            print(f"  id={i} [{tag}]  gold={g}")
            resp_a = textwrap.shorten((a[i].get("response") or "").replace("\n", " "), 120)
            resp_b = textwrap.shorten((b[i].get("response") or "").replace("\n", " "), 120)
            print(f"    baseline: {resp_a}")
            print(f"    new:      {resp_b}")

    preview(gains, "gains")
    preview(regressions, "regressions")


if __name__ == "__main__":
    main()
