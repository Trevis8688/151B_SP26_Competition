"""
exp_034 — SFT v2 data prep.

Pulls NuminaMath-CoT (FF) + MathQA (MCQ) from HF Hub, wraps both into a unified
chat-format JSONL that train_sft.py consumes. Probe split is the first N
examples of the shuffled train mix, so resuming from the probe adapter
continues training naturally without re-processing.

Output:
  /tmp/sft_data/train.jsonl   ~7000 examples (5000 FF + 2000 MCQ, shuffled)
  /tmp/sft_data/probe.jsonl   first 500 lines of train.jsonl (subset)

Each line:
  {"messages": [{"role": "user", "content": ...},
                {"role": "assistant", "content": "<think>{rationale}</think>\n\n\\boxed{answer}"}],
   "source": "numina" | "mathqa"}
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

from datasets import load_dataset


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _extract_last_boxed(text: str) -> str | None:
    """Pull the last \\boxed{...} content from a NuminaMath solution. None if absent."""
    matches = BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def _strip_last_boxed(text: str) -> str:
    """Remove the trailing '\\boxed{...}' (and any 'Therefore.../So.../=' lead-in punctuation)
    from a solution so it can serve as the <think> rationale.

    Conservative: only strips one trailing boxed expression. If the solution has many
    boxed expressions, only the last is removed.
    """
    matches = list(BOXED_RE.finditer(text or ""))
    if not matches:
        return text
    last = matches[-1]
    return text[: last.start()].rstrip(" \t.\n=:")


def prep_numina(n: int, seed: int, max_seq_len: int) -> list[dict]:
    print(f"[numina] loading AI-MO/NuminaMath-CoT (streaming), target n={n}")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    out: list[dict] = []
    skipped_no_boxed = 0
    skipped_too_long = 0
    seen = 0
    for ex in ds:
        seen += 1
        if seen % 5000 == 0:
            print(f"[numina] scanned {seen}, kept {len(out)}")
        problem = (ex.get("problem") or "").strip()
        solution = (ex.get("solution") or "").strip()
        if not problem or not solution:
            continue
        answer = _extract_last_boxed(solution)
        if answer is None:
            skipped_no_boxed += 1
            continue
        rationale = _strip_last_boxed(solution).strip()
        if not rationale:
            continue

        assistant = f"<think>{rationale}</think>\n\n\\boxed{{{answer}}}"
        approx_tokens = (len(problem) + len(assistant)) // 4  # cheap upper bound
        if approx_tokens > max_seq_len:
            skipped_too_long += 1
            continue

        out.append({
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": assistant},
            ],
            "source": "numina",
        })
        if len(out) >= n:
            break

    print(f"[numina] kept {len(out)} / scanned {seen} "
          f"(skipped no_boxed={skipped_no_boxed}, too_long={skipped_too_long})")
    return out


_OPT_RE = re.compile(r"\s*([a-e])\s*\)\s*(.*?)(?=\s*,\s*[a-e]\s*\)|$)", re.IGNORECASE | re.DOTALL)


def _parse_mathqa_options(options_str: str) -> list[tuple[str, str]]:
    """MathQA 'options' field is 'a ) 1.20 % , b ) 1.30 % , c ) 1.40 % , d ) 1.50 % , e ) 1.60 %'.
    Returns list of (letter_lower, text). Letters are guaranteed to be a..e.
    """
    return [(m.group(1).lower(), m.group(2).strip().rstrip(","))
            for m in _OPT_RE.finditer(options_str or "")]


def prep_mathqa(n: int, seed: int, max_seq_len: int) -> list[dict]:
    print(f"[mathqa] loading allenai/math_qa (streaming), target n={n}")
    ds = load_dataset("allenai/math_qa", split="train", streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=seed + 1, buffer_size=10000)

    out: list[dict] = []
    skipped_parse = 0
    skipped_too_long = 0
    seen = 0
    for ex in ds:
        seen += 1
        if seen % 2000 == 0:
            print(f"[mathqa] scanned {seen}, kept {len(out)}")
        problem = (ex.get("Problem") or "").strip()
        options_str = (ex.get("options") or "").strip()
        rationale = (ex.get("Rationale") or "").strip()
        correct = (ex.get("correct") or "").strip().lower()
        if not problem or not options_str or not rationale or correct not in "abcde":
            skipped_parse += 1
            continue
        opts = _parse_mathqa_options(options_str)
        if len(opts) < 2:
            skipped_parse += 1
            continue
        # Map to A..E uppercase in the user-facing question (matches competition MCQ style),
        # but MathQA's 'correct' is lowercase a..e. Keep the answer letter as the lowercase
        # letter from MathQA — the inference rubric is case-insensitive for the letter.
        user_msg_parts = [problem, "", "Options:"]
        for letter, text in opts:
            user_msg_parts.append(f"{letter.upper()}. {text}")
        user_msg = "\n".join(user_msg_parts)
        # Answer letter: use uppercase to match the competition format extracted by judger
        answer_letter = correct.upper()
        assistant = f"<think>{rationale}</think>\n\n\\boxed{{{answer_letter}}}"
        approx_tokens = (len(user_msg) + len(assistant)) // 4
        if approx_tokens > max_seq_len:
            skipped_too_long += 1
            continue

        out.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant},
            ],
            "source": "mathqa",
        })
        if len(out) >= n:
            break

    print(f"[mathqa] kept {len(out)} / scanned {seen} "
          f"(skipped parse={skipped_parse}, too_long={skipped_too_long})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/tmp/sft_data")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.json"))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())["data"]
    ff_n = int(cfg["ff_n"])
    mcq_n = int(cfg["mcq_n"])
    probe_n = int(cfg["probe_n"])
    seed = int(cfg["seed"])
    max_seq_len = int(cfg["max_seq_len"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    numina = prep_numina(ff_n, seed, max_seq_len)
    mathqa = prep_mathqa(mcq_n, seed, max_seq_len)

    all_examples = numina + mathqa
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    print(f"[mix] total={len(all_examples)} (numina={len(numina)}, mathqa={len(mathqa)})")

    train_path = out_dir / "train.jsonl"
    with train_path.open("w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[write] {train_path}  ({len(all_examples)} lines)")

    probe_path = out_dir / "probe.jsonl"
    probe = all_examples[:probe_n]
    with probe_path.open("w") as f:
        for ex in probe:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[write] {probe_path}  ({len(probe)} lines, source mix:"
          f" numina={sum(1 for e in probe if e['source']=='numina')},"
          f" mathqa={sum(1 for e in probe if e['source']=='mathqa')})")


if __name__ == "__main__":
    main()
