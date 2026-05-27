"""Build the 150-sample probe set for exp_036 from exp_018's public_responses.scored.jsonl.

Stratified sample:
  - 50 wrong_math FF  (the target population — verifier must flag these)
  - 50 correct FF     (the control — verifier must NOT flag these)
  - 50 correct MCQ    (secondary control — verifier must NOT flag these)

For each sample we extract the model's PROPOSED answer (last \boxed{} content)
and write a row with the verification prompt ready to feed vLLM:
  {id, segment, ground_truth_correct, proposed_answer, messages}

Output: experiments/exp_036_verification_probe/probe_samples.jsonl
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def last_boxed(text: str) -> str:
    matches = BOX_RE.findall(text or "")
    return matches[-1].strip() if matches else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored",
                    default=str(REPO_ROOT / "experiments/exp_018_pass2_rescue/public_responses.scored.jsonl"))
    ap.add_argument("--public", default=str(REPO_ROOT / "data/public.jsonl"))
    ap.add_argument("--out",
                    default=str(Path(__file__).parent / "probe_samples.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_wrong_ff", type=int, default=50)
    ap.add_argument("--n_correct_ff", type=int, default=50)
    ap.add_argument("--n_correct_mcq", type=int, default=50)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from prompts import build_verify_messages

    pub = {json.loads(l)["id"]: json.loads(l) for l in open(args.public)}
    rows = [json.loads(l) for l in open(args.scored)]

    # Bucket the public into three pools
    pool_wrong_ff, pool_correct_ff, pool_correct_mcq = [], [], []
    for r in rows:
        q = pub[r["id"]]
        is_mcq = bool(q.get("options"))
        proposed = last_boxed(r.get("response", ""))
        if not proposed:
            continue  # missing_boxed; nothing for the verifier to operate on
        item = {
            "id": r["id"],
            "is_mcq": is_mcq,
            "ground_truth_correct": bool(r["correct"]),
            "proposed_answer": proposed,
            "gold": q["answer"],
            "question": q["question"],
            "options": q.get("options"),
        }
        if not is_mcq and not r["correct"]:
            pool_wrong_ff.append(item)
        elif not is_mcq and r["correct"]:
            pool_correct_ff.append(item)
        elif is_mcq and r["correct"]:
            pool_correct_mcq.append(item)

    rng = random.Random(args.seed)
    rng.shuffle(pool_wrong_ff); rng.shuffle(pool_correct_ff); rng.shuffle(pool_correct_mcq)
    sampled = (pool_wrong_ff[:args.n_wrong_ff]
               + pool_correct_ff[:args.n_correct_ff]
               + pool_correct_mcq[:args.n_correct_mcq])
    print(f"Pools available: wrong_ff={len(pool_wrong_ff)} correct_ff={len(pool_correct_ff)} correct_mcq={len(pool_correct_mcq)}")
    print(f"Sampled: wrong_ff={args.n_wrong_ff} correct_ff={args.n_correct_ff} correct_mcq={args.n_correct_mcq} (total={len(sampled)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in sampled:
            messages = build_verify_messages(s["question"], s["proposed_answer"], s["is_mcq"])
            row = {
                "id": s["id"],
                "is_mcq": s["is_mcq"],
                "ground_truth_correct": s["ground_truth_correct"],
                "proposed_answer": s["proposed_answer"],
                "gold": s["gold"],
                "messages": messages,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
