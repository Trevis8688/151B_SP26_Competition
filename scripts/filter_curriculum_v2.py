"""Filter difficulty_samples_v2.jsonl into a curriculum for exp_015 GRPO pass 2.

The sampler (sample_difficulty_v2.py) builds a default curriculum at the end of
its run with the strict criterion (1<=num_correct<=3 AND num_clipped==0). This
script lets us re-filter the same JSONL with different knobs *without* re-running
sampling -- useful for sensitivity analysis or for shaping the MCQ/free-form mix.

Usage:
    python scripts/filter_curriculum_v2.py
        --in  data/difficulty_samples_v2.jsonl
        --out experiments/exp_015_grpo_pass2/curriculum_v2.json
        --min-correct 1 --max-correct 3
        [--allow-clipped]
        [--ff-mcq-ratio 2.0]   # cap MCQ count to len(ff)/ratio
"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_samples(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_rows(rows, min_correct, max_correct, allow_clipped):
    keep = []
    for r in rows:
        nc = r["num_correct"]
        if not (min_correct <= nc <= max_correct):
            continue
        if (not allow_clipped) and r["num_clipped"] > 0:
            continue
        keep.append(r)
    return keep


def cap_mcq_ratio(rows, ff_mcq_ratio, seed=42):
    mcq = [r for r in rows if r["is_mcq"]]
    ff  = [r for r in rows if not r["is_mcq"]]
    cap = int(len(ff) / ff_mcq_ratio)
    if len(mcq) <= cap:
        return rows
    rng = random.Random(seed)
    rng.shuffle(mcq)
    mcq = mcq[:cap]
    return mcq + ff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  default=str(REPO / "data" / "difficulty_samples_v2.jsonl"))
    ap.add_argument("--out", dest="outp", default=str(REPO / "experiments" / "exp_015_grpo_pass2" / "curriculum_v2.json"))
    ap.add_argument("--min-correct", type=int, default=1)
    ap.add_argument("--max-correct", type=int, default=3)
    ap.add_argument("--allow-clipped", action="store_true",
                    help="Keep prompts even if some of the 4 samples were truncated.")
    ap.add_argument("--ff-mcq-ratio", type=float, default=None,
                    help="If set, cap MCQ count so free-form:MCQ >= this ratio.")
    args = ap.parse_args()

    inp  = Path(args.inp)
    outp = Path(args.outp)
    if not inp.exists():
        sys.exit(f"ERR: input not found: {inp}")
    outp.parent.mkdir(parents=True, exist_ok=True)

    rows = load_samples(inp)
    print(f"Loaded {len(rows)} sampled prompts from {inp}")

    nc_dist = Counter(r["num_correct"] for r in rows)
    clip_total = sum(1 for r in rows if r["num_clipped"] > 0)
    print(f"num_correct distribution: " + "  ".join(f"{k}={nc_dist.get(k,0)}" for k in range(5)))
    print(f"Prompts with >=1 clipped sample: {clip_total}/{len(rows)}")

    kept = filter_rows(rows, args.min_correct, args.max_correct, args.allow_clipped)
    mcq_n = sum(r["is_mcq"] for r in kept)
    ff_n  = len(kept) - mcq_n
    print(f"After filter (correct in [{args.min_correct},{args.max_correct}], "
          f"allow_clipped={args.allow_clipped}): {len(kept)} prompts ({mcq_n} MCQ, {ff_n} FF)")

    if args.ff_mcq_ratio is not None:
        kept = cap_mcq_ratio(kept, args.ff_mcq_ratio)
        mcq_n = sum(r["is_mcq"] for r in kept)
        ff_n  = len(kept) - mcq_n
        print(f"After FF:MCQ ratio cap >= {args.ff_mcq_ratio}: {len(kept)} prompts "
              f"({mcq_n} MCQ, {ff_n} FF)")

    sweet_ids = sorted(r["id"] for r in kept)
    payload = {
        "sweet_ids": sweet_ids,
        "n_sweet": len(sweet_ids),
        "breakdown": {"mcq": mcq_n, "ff": ff_n},
        "source": str(inp.relative_to(REPO)),
        "filter": {
            "min_correct": args.min_correct,
            "max_correct": args.max_correct,
            "allow_clipped": args.allow_clipped,
            "ff_mcq_ratio": args.ff_mcq_ratio,
        },
        "num_correct_distribution": {str(k): nc_dist.get(k, 0) for k in range(5)},
    }
    with open(outp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote curriculum: {outp}")
    print(f"  {len(sweet_ids)} IDs  ({mcq_n} MCQ, {ff_n} FF)")


if __name__ == "__main__":
    main()
