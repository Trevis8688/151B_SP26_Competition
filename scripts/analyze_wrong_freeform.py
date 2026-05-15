"""Categorize wrong free-form failures from a results.jsonl into math domains.

Why: exp_014 has 329 wrong free-form cases with \\boxed present (true reasoning
failures, not truncation). Knowing whether they cluster in 1-2 math domains
shapes every downstream lever — targeted few-shots, SFT data shopping list,
domain-specific rescue prompts.

Approach: priority-first keyword/regex matching against the *question* text
(joined in from public.jsonl). First matching bucket wins, so order matters —
narrower buckets (calculus, linalg) come before broader ones (algebra, word).

Output:
  - stdout: bucket counts + 3 example IDs per bucket
  - --json out.json: full per-bucket assignments + IDs (for follow-up scripts)

Usage:
    python scripts/analyze_wrong_freeform.py \\
        --results experiments/exp_014_rescue_v2_grpo/results.jsonl \\
        --json reports/exp_014_wrong_ff_buckets.json
"""
import argparse, json, re, sys
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
PUBLIC = REPO / "data" / "public.jsonl"


def has_any(text, *patterns):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


# Priority-ordered. Each tuple: (bucket name, list of regex patterns).
# Narrower / more specific buckets first.
BUCKETS = [
    ("applied_modeling", [
        r"\b(half[- ]?life|doubling time)\b",
        r"\b(exponential|logarithmic) (decay|growth|function|model)\b",
        r"\bcompound(ed)? interest\b", r"\bcontinuously compounded\b",
        r"\bradioactive\b", r"\b(carbon|isotope)\b.*\bdecay\b",
        r"\bpopulation (model|growth|of)\b", r"\bbacteria\b",
        r"\bNewton'?s law of cooling\b", r"\b(cooling|heating)\b.*\btemperature\b",
        r"\b(degrees? )?(fahrenheit|celsius|kelvin)\b",
        r"\b(percent|percentage)\b.*\b(increase|decrease|change|of)\b",
        r"\bmodel(s|ed|ing)?\b.*\bby\b.*=",
        r"\bappreciat(ed|ion)\b", r"\bdepreciat(ed|ion)\b",
        r"\bcontinuous rate\b", r"\bgrowing at\b.*\brate\b",
        r"\bdecibel\b", r"\bRichter\b", r"\bpH\b",
    ]),
    ("calculus", [
        r"\bderivative\b", r"\bintegral\b", r"\bintegrate\b", r"\\int\b",
        r"\bdy\s*/\s*dx\b", r"\bd/dx\b", r"\blim(it)?\b.*\\to", r"\\frac\{d",
        r"\bantiderivative\b", r"\bMaclaurin\b", r"\bTaylor\b",
        r"\bdifferent(ial|iate)\b",
    ]),
    ("linear_algebra", [
        r"\bmatri(x|ces)\b", r"\bdeterminant\b", r"\beigen(value|vector)\b",
        r"\bdot product\b", r"\bcross product\b", r"\\mathbf\{", r"\bvector\b",
        r"\brank\b", r"\bnull space\b", r"\bbasis\b",
    ]),
    ("trigonometry", [
        r"\bsin\(", r"\bcos\(", r"\btan\(", r"\\sin\b", r"\\cos\b", r"\\tan\b",
        r"\bradian", r"\barc(sin|cos|tan)\b",
    ]),
    ("number_theory", [
        r"\bprime\b", r"\bdivisible\b", r"\bdivisor", r"\bgcd\b", r"\blcm\b",
        r"\b(modulo|mod\b|\\pmod)\b", r"\bfactorial\b", r"\bdigits?\b.*\b(sum|number|how many)\b",
        r"\binteger solutions\b", r"\bperfect (square|cube)\b", r"\bremainder\b",
    ]),
    ("combinatorics_probability", [
        r"\bprobabil", r"\\binom\{", r"\bchoose\b", r"\bcombination",
        r"\bpermutation", r"\bexpected (value|number)\b", r"\barrangement",
        r"\bhow many ways\b", r"\bdice\b", r"\bcoin\b.*\b(flip|toss)\b",
    ]),
    ("statistics", [
        r"\bmean\b", r"\bmedian\b", r"\bmode\b", r"\baverage\b",
        r"\bvariance\b", r"\bstandard deviation\b", r"\bstd\.? dev",
        r"\bhypothes(is|es)\b", r"H_0\b", r"H_A\b", r"H_a\b",
        r"\bp[- ]?value\b", r"\bconfidence interval\b", r"\bsignificance level\b",
        r"\b(t|z|F|chi[- ]?square)[- ]?(test|statistic|distribution|value|curve)\b",
        r"\bdegrees? of freedom\b", r"\bdf\s*=", r"\bnormal distribution\b",
        r"\b(sample|population) (mean|proportion|variance|size|standard)\b",
        r"\b(null|alternative) hypothesis\b", r"\bregression\b",
        r"\bcorrelation\b", r"\b(margin of error)\b",
        r"\b(percent(ile)?|quartile)\b",
    ]),
    ("sequences_series", [
        r"\b(arithmetic|geometric) (sequence|series|progression)\b",
        r"\bn(th| -?th) term\b", r"\brecurrence\b", r"\brecursive\b", r"\bFibonacci\b",
        r"\bsum of (the )?(first|infinite|geometric|arithmetic)\b",
        r"\bsequence\b", r"\bseries\b", r"\\sum_",
        r"\bfirst (four|five|three|n) terms\b", r"\bbinomial expansion\b",
    ]),
    ("geometry", [
        r"\btriangle\b", r"\bcircle\b", r"\bsquare\b", r"\brectangle\b",
        r"\bpolygon\b", r"\barea of\b", r"\bvolume of\b", r"\bperimeter\b",
        r"\bcircumference\b", r"\bradius\b", r"\bdiameter\b", r"\bangle\b",
        r"\bpolyhedr", r"\bsphere\b", r"\bcylinder\b", r"\bcone\b",
        r"\bparallelogram\b", r"\bhexagon\b", r"\bpentagon\b", r"\boctagon\b",
        r"\baxis\b.*\bsymmetry\b", r"\bcoordinates?\b.*\bplane\b",
    ]),
    ("algebra", [
        r"\bsolve\b.*=\s*", r"\bequation\b", r"\bpolynomial\b", r"\binequality\b",
        r"\bquadratic\b", r"\bsystem of (linear )?equation",
        r"\bfunction f\b", r"\blogarithm\b", r"\b\\log\b", r"\\sqrt\{",
        r"x\^2", r"x\^\{2\}",
    ]),
]


def categorize(question: str):
    for name, pats in BUCKETS:
        if has_any(question, *pats):
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--json", default=None, help="Optional output JSON path.")
    ap.add_argument("--examples-per-bucket", type=int, default=3)
    ap.add_argument("--show-gold", action="store_true",
                    help="Print the gold answer alongside each example ID.")
    args = ap.parse_args()

    rpath = Path(args.results).resolve()
    if not rpath.exists():
        sys.exit(f"ERR: {rpath} not found")

    questions = {}
    with open(PUBLIC) as f:
        for line in f:
            r = json.loads(line)
            questions[r["id"]] = r["question"]

    wrong_ids = []
    gold_by_id = {}
    with open(rpath) as f:
        for line in f:
            r = json.loads(line)
            if r["correct"]:
                continue
            if r["is_mcq"]:
                continue
            if "\\boxed" not in r["response"]:
                continue
            wrong_ids.append(r["id"])
            gold_by_id[r["id"]] = r["gold"]

    print(f"Loaded {len(wrong_ids)} wrong free-form (boxed-present) cases from {rpath}")
    print()

    buckets = defaultdict(list)
    for qid in wrong_ids:
        q = questions.get(qid, "")
        buckets[categorize(q)].append(qid)

    counts = Counter({k: len(v) for k, v in buckets.items()})
    total = sum(counts.values())
    width = max(len(k) for k in counts)

    print(f"{'bucket':<{width}}  count   pct")
    print("-" * (width + 16))
    for name, n in counts.most_common():
        print(f"{name:<{width}}  {n:>5}  {100*n/total:>5.1f}%")
    print("-" * (width + 16))
    print(f"{'TOTAL':<{width}}  {total:>5}  100.0%")
    print()

    print("Sample IDs per bucket (truncated to first 100 chars of question):")
    print()
    for name, _ in counts.most_common():
        print(f"=== {name} ({len(buckets[name])}) ===")
        for qid in buckets[name][: args.examples_per_bucket]:
            q = questions.get(qid, "").replace("\n", " ").strip()
            head = q[:100] + ("..." if len(q) > 100 else "")
            line = f"  id={qid}: {head}"
            if args.show_gold:
                line += f"  gold={gold_by_id[qid]}"
            print(line)
        print()

    if args.json:
        outp = Path(args.json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(rpath.relative_to(REPO)),
            "n_wrong_ff_boxed": total,
            "counts": dict(counts),
            "buckets": {name: sorted(ids) for name, ids in buckets.items()},
        }
        with open(outp, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
