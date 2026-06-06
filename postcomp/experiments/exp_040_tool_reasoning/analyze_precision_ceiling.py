#!/usr/bin/env python3
"""Measure the precision-recoverable ceiling on dev free-form errors (GPU-free).

Turns the hand-estimated "~25-35% of FF errors are precision" into hard numbers by
re-judging through the REAL auto_judge path, under two simulated interventions:

  (A) PROMPT-ONLY "don't round": for each numeric token the model boxed, find the
      higher-precision number it ALREADY wrote in its trace whose rounding equals the
      boxed value, and substitute it. This simulates "same reasoning, reported at full
      precision" — exactly what the prompt-only intervention does. Non-numeric tokens
      (letters/expressions) are left as the model had them.

  (B) TOOL estimate (upper-ish bound): additionally credit cases where the model wrote
      an accurate-but-imprecise value (relerr in (1e-8, 1e-4]) that an exact tool would
      nail. This brackets what one-shot PAL can recover above prompt-only.

Run: ~/miniconda3/envs/my-virtenv/bin/python \
        postcomp/experiments/exp_040_tool_reasoning/analyze_precision_ceiling.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from judger import Judger  # noqa: E402

J = Judger(strict_extract=False)

DEV = REPO / "data" / "splits" / "dev.jsonl"
SCORED = REPO / "experiments" / "exp_018_pass2_rescue" / "results.newjudge.jsonl"

_NUM = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)?")


def real_judge(boxed_tokens, gold_list) -> bool:
    pred = "\\boxed{" + ", ".join(str(t) for t in boxed_tokens) + "}"
    try:
        return bool(J.auto_judge(pred=pred, gold=gold_list, options=[[]] * len(gold_list)))
    except Exception:
        return False


def as_float(tok):
    t = str(tok).strip().strip("$").replace("\\%", "").replace("%", "").replace(",", "")
    try:
        return float(t)
    except Exception:
        return None


def decimals(tok):
    m = re.search(r"\.(\d+)", str(tok))
    return len(m.group(1)) if m else 0


def trace_numbers(text):
    """All numeric literals in the response, as (float, raw_string)."""
    out = []
    for m in _NUM.finditer(text or ""):
        v = as_float(m.group(0))
        if v is not None:
            out.append((v, m.group(0)))
    return out


def extract_boxed(text):
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + 7
    depth, out = 1, []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None


def box_tokens(boxed):
    return [t.strip() for t in boxed.split(",") if t.strip()] if boxed else []


def unrounded_from_trace(boxed_tok, trace):
    """Higher-precision in-trace number whose rounding == the boxed value, else None."""
    bv = as_float(boxed_tok)
    if bv is None:
        return None
    nd = decimals(boxed_tok)
    best = None
    for v, raw in trace:
        if round(v, nd) == round(bv, nd) and decimals(raw) > nd:
            if best is None or decimals(raw) > decimals(best):
                best = raw
    return best


def main():
    dev = [json.loads(l) for l in open(DEV)]
    dev_ff = {d["id"] for d in dev if not d.get("options")}
    scored = {r["id"]: r for r in (json.loads(l) for l in open(SCORED))}
    errors = [scored[i] for i in dev_ff if not scored[i]["correct"]]

    a_flips, b_extra = [], []
    for r in errors:
        gold = r["gold"] if isinstance(r["gold"], list) else [r["gold"]]
        resp = r["response"] or ""
        mt = box_tokens(extract_boxed(resp))
        trace = trace_numbers(resp)
        if len(mt) != len(gold):
            continue  # multi-part/format mismatch — not a pure precision case

        # (A) substitute unrounded in-trace values
        sub = []
        changed = False
        for tok in mt:
            ur = unrounded_from_trace(tok, trace)
            if ur is not None:
                sub.append(ur)
                changed = True
            else:
                sub.append(tok)
        if changed and real_judge(sub, gold):
            a_flips.append(r["id"])
            continue

        # (B) tool estimate: model wrote an accurate-but-imprecise value a tool would fix
        recoverable = True
        for i, g in enumerate(gold):
            gv = as_float(g)
            if gv is None:
                recoverable = False  # non-numeric gold — out of (B)'s numeric scope here
                break
            toks = [v for v, _ in trace]
            rel = min((abs((v - gv) / gv) if gv else abs(v)) for v in toks) if toks else 9e9
            if rel > 1e-4:
                recoverable = False
                break
        if recoverable:
            b_extra.append(r["id"])

    n = len(errors)
    print(f"dev free-form errors (exp_018, new judge): {n}\n")
    print(f"(A) PROMPT-ONLY 'don't round' — flips via model's own in-trace precision:")
    print(f"    {len(a_flips)}/{n} = {len(a_flips)/n:.0%}   ids={sorted(a_flips)}\n")
    print(f"(B) TOOL adds (accurate-but-imprecise hand value an exact tool would fix):")
    print(f"    +{len(b_extra)}/{n}   ids={sorted(b_extra)}")
    total = len(a_flips) + len(b_extra)
    print(f"\nCombined precision/compute ceiling (A+B): {total}/{n} = {total/n:.0%}")
    print(f"   -> recovering these lifts dev FF acc from "
          f"{(100-n)/100:.0%} toward {(100-n+total)/100:.0%}")
    print("\nNote: this is the *recoverable-by-precision* ceiling. The remaining errors are")
    print("multi-part/format (out of scope), truncation, or genuine reasoning failures.")


if __name__ == "__main__":
    main()
