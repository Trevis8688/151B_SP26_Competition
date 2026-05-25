"""
exp_034 — SFT v2 data prep.

Pulls NuminaMath-CoT (FF) + AQuA-RAT (MCQ) from HF Hub, wraps both into TRL
0.21's conversational prompt-completion format that train_sft.py consumes.
Probe split is the first N examples of the shuffled train mix, so resuming
from the probe adapter continues training naturally without re-processing.

MCQ source = `deepmind/aqua_rat` (swapped from `allenai/math_qa` on 2026-05-24:
math_qa ships a loading script, which the current `datasets` lib refuses;
aqua_rat is natively parquet, ~97k rows, same domain — MathQA was derived
from AQuA-RAT).

Output:
  /tmp/sft_data/train.jsonl   ~7000 examples (5000 FF + 2000 MCQ, shuffled)
  /tmp/sft_data/probe.jsonl   first 500 lines of train.jsonl (subset)

Each line (TRL conversational prompt-completion format):
  {"prompt": [{"role": "system", "content": ...},
              {"role": "user",   "content": ...}],
   "completion": [{"role": "assistant",
                   "content": "<think>{rationale}</think>\n\n\\boxed{answer}"}],
   "source": "numina" | "aqua"}
"""

import argparse
import importlib.util
import json
import os
import random
import re
from pathlib import Path

from datasets import load_dataset


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

# System prompts come from exp_017 so SFT examples match the inference prompt
# format the model will see at test time (in-distribution). Few-shots are NOT
# included per-example (they'd bloat every sequence); the system prompt is the
# load-bearing part for format matching.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_P_SPEC = importlib.util.spec_from_file_location(
    "exp017_prompts", str(_REPO_ROOT / "experiments/exp_017_pass2_stage1/prompts.py"))
_P = importlib.util.module_from_spec(_P_SPEC)
_P_SPEC.loader.exec_module(_P)
SYSTEM_PROMPT_MATH = _P.SYSTEM_PROMPT_MATH
SYSTEM_PROMPT_MCQ = _P.SYSTEM_PROMPT_MCQ


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
        approx_tokens = (len(SYSTEM_PROMPT_MATH) + len(problem) + len(assistant)) // 4
        if approx_tokens > max_seq_len:
            skipped_too_long += 1
            continue

        out.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_MATH},
                {"role": "user", "content": problem},
            ],
            "completion": [{"role": "assistant", "content": assistant}],
            "source": "numina",
        })
        if len(out) >= n:
            break

    print(f"[numina] kept {len(out)} / scanned {seen} "
          f"(skipped no_boxed={skipped_no_boxed}, too_long={skipped_too_long})")
    return out


_AQUA_OPT_RE = re.compile(r"^\s*([A-E])\s*\)\s*(.*)$", re.DOTALL)


def _parse_aqua_option(opt: str) -> tuple[str, str] | None:
    """AQuA-RAT options are list entries like 'A)21' or 'B) 1.30 %'.
    Returns (uppercase_letter, text) or None if it doesn't match.
    """
    m = _AQUA_OPT_RE.match(opt or "")
    if not m:
        return None
    return m.group(1).upper(), m.group(2).strip()


def prep_aqua(n: int, seed: int, max_seq_len: int) -> list[dict]:
    """AQuA-RAT (deepmind/aqua_rat) replaces allenai/math_qa.
    Swapped 2026-05-24: math_qa ships a loading script, current `datasets` refuses.
    AQuA-RAT is natively parquet, 97k train rows, same domain (MathQA is derived
    from AQuA-RAT). Schema: question, options=list[str], rationale, correct (A..E).
    """
    print(f"[aqua] loading deepmind/aqua_rat raw/train (streaming), target n={n}")
    ds = load_dataset("deepmind/aqua_rat", "raw", split="train", streaming=True)
    ds = ds.shuffle(seed=seed + 1, buffer_size=10000)

    out: list[dict] = []
    skipped_parse = 0
    skipped_too_long = 0
    seen = 0
    for ex in ds:
        seen += 1
        if seen % 2000 == 0:
            print(f"[aqua] scanned {seen}, kept {len(out)}")
        problem = (ex.get("question") or "").strip()
        options = ex.get("options") or []
        rationale = (ex.get("rationale") or "").strip()
        correct = (ex.get("correct") or "").strip().upper()
        if not problem or not options or not rationale or correct not in "ABCDE":
            skipped_parse += 1
            continue
        parsed = [_parse_aqua_option(o) for o in options]
        if any(p is None for p in parsed) or len(parsed) < 2:
            skipped_parse += 1
            continue

        user_msg_parts = [problem, "", "Options:"]
        for letter, text in parsed:
            user_msg_parts.append(f"{letter}. {text}")
        user_msg = "\n".join(user_msg_parts)

        assistant = f"<think>{rationale}</think>\n\n\\boxed{{{correct}}}"
        approx_tokens = (len(SYSTEM_PROMPT_MCQ) + len(user_msg) + len(assistant)) // 4
        if approx_tokens > max_seq_len:
            skipped_too_long += 1
            continue

        out.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_MCQ},
                {"role": "user", "content": user_msg},
            ],
            "completion": [{"role": "assistant", "content": assistant}],
            "source": "aqua",
        })
        if len(out) >= n:
            break

    print(f"[aqua] kept {len(out)} / scanned {seen} "
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
    aqua = prep_aqua(mcq_n, seed, max_seq_len)

    all_examples = numina + aqua
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    print(f"[mix] total={len(all_examples)} (numina={len(numina)}, aqua={len(aqua)})")

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
          f" aqua={sum(1 for e in probe if e['source']=='aqua')})")


if __name__ == "__main__":
    main()
