"""Deterministic dev/test split of public.jsonl for local iteration.

Dev = small stratified subset for fast iteration (~20 min on Kaggle).
Test = everything else for mid-cycle validation.

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --dev-size 200 --seed 42
"""
import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/public.jsonl")
    p.add_argument("--out-dir", default="data/splits")
    p.add_argument("--dev-size", type=int, default=200,
                   help="Total dev questions (half MCQ, half free-form).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = [json.loads(line) for line in open(args.data)]
    mcq = [d for d in data if d.get("options")]
    free = [d for d in data if not d.get("options")]

    rng = random.Random(args.seed)
    rng.shuffle(mcq)
    rng.shuffle(free)

    half = args.dev_size // 2
    dev = mcq[:half] + free[:half]
    test = mcq[half:] + free[half:]

    dev_ids = {d["id"] for d in dev}
    assert len(dev_ids) == len(dev), "dev contains duplicate ids"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, subset in [("dev", dev), ("test", test)]:
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for d in subset:
                f.write(json.dumps(d) + "\n")
        n_mcq = sum(bool(d.get("options")) for d in subset)
        n_free = len(subset) - n_mcq
        print(f"  {name:4s}: {len(subset):4d} questions  ({n_mcq} MCQ, {n_free} free-form)  → {path}")


if __name__ == "__main__":
    main()
