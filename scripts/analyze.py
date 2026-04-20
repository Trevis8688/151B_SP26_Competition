"""Bucket errors in a results.jsonl into actionable failure categories.

Reads a results file produced by the starter notebook (fields: id, is_mcq,
gold, response, correct), prints overall accuracy, breakdown by MCQ vs
free-form, error-mode counts (missing \\boxed{}, wrong letter, wrong math),
and accuracy per topic.

Optionally filters to a split (dev / test) so the numbers are comparable
across experiments.

Usage:
    python scripts/analyze.py results.jsonl
    python scripts/analyze.py results.jsonl --split dev
    python scripts/analyze.py results.jsonl --json   # machine-readable
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


TOPIC_KEYWORDS = {
    "calculus":        ["integral", "derivative", "differentiat", "limit", "lim ", "d/dx", "antiderivat", "∫"],
    "linear_algebra":  ["matrix", "matrices", "eigenvalue", "eigenvector", "determinant", "vector space", "linear transformation"],
    "probability":     ["probability", "expected value", "random variable", "distribution", "combinat", "permut", "choose"],
    "algebra":         ["polynomial", "equation", "solve for", "factor", "root", "quadratic", "inequality"],
    "geometry":        ["triangle", "circle", "area", "perimeter", "angle", "polygon", "radius", "diameter"],
    "number_theory":   ["divisib", "prime", "modulo", "mod ", "gcd", "lcm", "congruen", "remainder"],
    "series":          ["series", "sequence", "convergent", "divergent", "sum of", "summation", "taylor", "power series"],
    "trigonometry":    ["sin", "cos", "tan", "trigonometr", "arcsin", "arccos", "arctan"],
    "differential_eq": ["differential equation", "ivp", "ode", "separable", "integrating factor"],
}


def detect_topics(question: str) -> list[str]:
    q = question.lower()
    topics = [t for t, kws in TOPIC_KEYWORDS.items() if any(kw in q for kw in kws)]
    return topics or ["other"]


def classify_error(r: dict) -> str:
    if r["correct"]:
        return "correct"
    has_boxed = "\\boxed" in (r.get("response") or "")
    if not has_boxed:
        return "missing_boxed"
    return "wrong_mcq" if r.get("is_mcq") else "wrong_math"


def accuracy(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0


def load_results(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def load_split_ids(split_path: str) -> set:
    return {json.loads(line)["id"] for line in open(split_path)}


def analyze(results: list[dict], questions_by_id: dict) -> dict:
    mcq = [r for r in results if r["is_mcq"]]
    free = [r for r in results if not r["is_mcq"]]

    error_buckets = Counter(classify_error(r) for r in results)

    topic_total = Counter()
    topic_correct = Counter()
    for r in results:
        q = questions_by_id.get(r["id"])
        if q is None:
            topics = ["other"]
        else:
            topics = detect_topics(q["question"])
        for t in topics:
            topic_total[t] += 1
            if r["correct"]:
                topic_correct[t] += 1

    topic_acc = {
        t: {
            "correct": topic_correct[t],
            "total": topic_total[t],
            "acc": topic_correct[t] / max(topic_total[t], 1) * 100,
        }
        for t in sorted(topic_total, key=lambda t: topic_correct[t] / max(topic_total[t], 1))
    }

    return {
        "total": len(results),
        "overall_acc": accuracy(results),
        "mcq": {"correct": sum(r["correct"] for r in mcq), "total": len(mcq), "acc": accuracy(mcq)},
        "free": {"correct": sum(r["correct"] for r in free), "total": len(free), "acc": accuracy(free)},
        "errors": dict(error_buckets),
        "by_topic": topic_acc,
    }


def print_summary(s: dict):
    print("=" * 56)
    print(f"OVERALL       {s['overall_acc']:5.2f}%   ({s['total']} questions)")
    print(f"  MCQ         {s['mcq']['acc']:5.2f}%   ({s['mcq']['correct']}/{s['mcq']['total']})")
    print(f"  Free-form   {s['free']['acc']:5.2f}%   ({s['free']['correct']}/{s['free']['total']})")
    print("=" * 56)
    print("ERROR BUCKETS")
    for bucket in ("correct", "missing_boxed", "wrong_mcq", "wrong_math"):
        n = s["errors"].get(bucket, 0)
        pct = n / max(s["total"], 1) * 100
        print(f"  {bucket:18s} {n:4d}  ({pct:5.1f}%)")
    print("=" * 56)
    print("ACCURACY BY TOPIC  (ordered worst → best)")
    for t, m in s["by_topic"].items():
        bar_len = int(m["acc"] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {t:18s} {m['correct']:4d}/{m['total']:4d}  {bar} {m['acc']:5.1f}%")
    print("=" * 56)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--data", default="data/public.jsonl",
                   help="Source jsonl for question text (for topic detection).")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text summary.")
    args = p.parse_args()

    results = load_results(args.results)
    questions_by_id = {d["id"]: d for d in (json.loads(l) for l in open(args.data))}

    if args.split != "all":
        split_path = f"data/splits/{args.split}.jsonl"
        if not Path(split_path).exists():
            raise SystemExit(f"{split_path} not found — run scripts/make_splits.py first.")
        ids = load_split_ids(split_path)
        results = [r for r in results if r["id"] in ids]
        print(f"[filtered to {args.split}: {len(results)} questions]\n")

    summary = analyze(results, questions_by_id)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
