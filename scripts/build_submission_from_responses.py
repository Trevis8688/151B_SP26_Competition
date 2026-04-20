"""Build a Kaggle submission.csv from a saved private_responses.jsonl.

Usage:
    python scripts/build_submission_from_responses.py \
        path/to/private_responses.jsonl \
        path/to/submission.csv

The leaderboard test set is private.jsonl only, so the submission must
contain exactly the 943 private question IDs (0..942). Public IDs do not
belong in the submission.
"""
import csv
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    rows = [json.loads(l) for l in open(in_path)]
    print(f"Loaded {len(rows)} responses from {in_path}")

    # Deduplicate by id (keep last write) in case of repeats
    by_id = {r["id"]: r["response"] for r in rows}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "response"])
        for qid in sorted(by_id.keys()):
            w.writerow([qid, by_id[qid]])

    print(f"Wrote {len(by_id)} rows to {out_path}")


if __name__ == "__main__":
    main()
