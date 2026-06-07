#!/usr/bin/env python3
"""Timeout-safe wrapper around the repo judger (`judger.Judger.auto_judge`).

WHY THIS EXISTS (a real robustness finding — see postcomp/DEVLOG.md 2026-06-07):
`auto_judge` calls sympy `simplify`/`equals`, which has **no internal timeout** and
can spin *forever* on a pathological symbolic comparison built from model output.
The exp_040 probe proved this the expensive way: all 160 generations finished, then
the judging loop hung silently for ~6h until the DSMLP pod hit its wall-clock kill —
losing every generation, because outputs were only written *after* the loop.

A `try/except` cannot save you here: an infinite loop is not an exception. The only
robust fix is to run each judge in a **fresh subprocess** and SIGKILL it on timeout.
`subprocess.run(..., timeout=...)` sends SIGKILL on POSIX, so it survives even a
C-level (gmpy2/mpmath) hang that would ignore SIGTERM / `multiprocessing.terminate()`.
Running a *subprocess* (fork+exec) rather than `multiprocessing` (fork-only) also
sidesteps the fork-from-CUDA fragility when this is called from a vLLM process.

This is the standard judge entry point for any pipeline that scores model output:
the probe, the full 200q dev run, and any submission build. Don't call
`auto_judge` directly on untrusted-shape model text again — route through here.

Run `python postcomp/harness/safe_judge.py` for self-tests (no GPU). The self-test
includes the GATE the advisor asked for: prove a deliberate `while True: pass`
worker is actually KILLED within the timeout budget before trusting this on DSMLP.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # …/151B_SP26_Competition (holds judger.py)

# Layer-1 poison pre-filter (the confirmed bug-097 remedy, ported from
# scripts/sample_difficulty_v2.py). The historical 6.5h hang was a runaway-repetition
# `\boxed{...}` string that wedged parse_latex inside Antlr's C runtime. Real math
# answers are short; reject oversized boxed content BEFORE sympy/antlr is ever invoked,
# so the (bounded but slower) subprocess judge is reserved for plausible answers.
MAX_BOXED_LEN = int(os.environ.get("MAX_BOXED_LEN", "300"))

# Worker: reads {"pred":..., "gold":[...]} as JSON on stdin, prints "1"/"0".
# Run with `python -I` (isolated: ignores $PYTHON*, user-site) so it can't inherit a
# contaminated PVC user-site; the venv's own site-packages (sympy, antlr4) remain.
_JUDGE_WORKER = (
    "import json, sys\n"
    f"sys.path.insert(0, {str(_REPO)!r})\n"
    "data = json.load(sys.stdin)\n"
    "gold = data['gold']\n"
    "if not isinstance(gold, list):\n"
    "    gold = [gold]\n"
    "try:\n"
    "    from judger import Judger\n"
    "    J = Judger(strict_extract=False)\n"
    "    ok = bool(J.auto_judge(pred=data['pred'], gold=gold, options=[[]] * len(gold)))\n"
    "except Exception:\n"
    "    ok = False\n"
    "sys.stdout.write('1\\n' if ok else '0\\n')\n"
)

# Test-only worker that never returns — used by the self-test to prove the SIGKILL
# path actually kills a hang within budget (the advisor's pre-DSMLP gate).
_HANG_WORKER = "while True:\n    pass\n"


def _last_boxed_content(text: str) -> str | None:
    """Brace-balanced content of the last `\\boxed{...}` (None if absent). Used only
    by the poison pre-filter — kept local so safe_judge has no harness dependencies."""
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
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


def _run(worker_code: str, payload: str, timeout: float, python_exe: str):
    """Run a worker subprocess feeding `payload` on stdin. Returns the CompletedProcess,
    or None if it timed out (subprocess.run SIGKILLs the child on TimeoutExpired)."""
    try:
        return subprocess.run(
            [python_exe, "-I", "-c", worker_code],
            input=payload, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None  # child was killed by subprocess.run's cleanup (SIGKILL on POSIX)


def judge_safe(pred: str, gold, timeout: float = 20.0, python_exe: str | None = None) -> dict:
    """Judge `pred` against `gold` with a hard wall-clock kill.

    Args:
        pred: full model response text (must contain a `\\boxed{}` to score).
        gold: ground-truth answer — a str or a list of strs (multi-value).
        timeout: seconds before the judge subprocess is SIGKILLed.
        python_exe: interpreter for the worker (default: current — must have sympy+antlr4).

    Returns:
        {"correct": bool, "timed_out": bool, "poison": bool}. A timeout or poison
        rejection counts as correct=False; both are flagged so the caller can log which
        id tripped it (a timeout id is the sympy-hang culprit; a poison id had oversized
        boxed content rejected before judging).
    """
    # Layer 1: reject runaway boxed content without ever invoking sympy/antlr.
    boxed = _last_boxed_content(pred or "")
    if boxed is not None and len(boxed) > MAX_BOXED_LEN:
        return {"correct": False, "timed_out": False, "poison": True}

    # Layer 2: judge in a fresh subprocess SIGKILLed after `timeout`.
    python_exe = python_exe or sys.executable
    gold_list = gold if isinstance(gold, list) else [gold]
    payload = json.dumps({"pred": pred, "gold": gold_list})
    proc = _run(_JUDGE_WORKER, payload, timeout, python_exe)
    if proc is None:
        return {"correct": False, "timed_out": True, "poison": False}
    out = (proc.stdout or "").strip().splitlines()
    return {"correct": bool(out) and out[-1].strip() == "1", "timed_out": False, "poison": False}


# ─────────────────────────────────────────────────────────────────────────────
# Self-tests — `python postcomp/harness/safe_judge.py`  (no GPU; needs sympy+antlr4)
# ─────────────────────────────────────────────────────────────────────────────
def _selftest() -> int:
    import time
    failures = []

    def check(name, cond):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            failures.append(name)

    print("safe_judge self-tests:")

    # GATE (advisor #1): a deliberately-hanging worker MUST be killed within budget.
    # This is the whole point — prove the SIGKILL timeout works before trusting DSMLP.
    t0 = time.time()
    proc = _run(_HANG_WORKER, "{}", timeout=2.0, python_exe=sys.executable)
    elapsed = time.time() - t0
    check("GATE: hanging worker is KILLED (returns None)", proc is None)
    check(f"GATE: kill happens within budget (took {elapsed:.1f}s, budget 2s+grace)", elapsed < 8.0)

    # Layer-1 poison pre-filter: oversized boxed content rejected fast, no subprocess.
    poison = r"\boxed{" + "1" * (MAX_BOXED_LEN + 100) + "}"
    t0 = time.time()
    r = judge_safe(poison, ["42"], timeout=20)
    check("poison: oversized \\boxed{} rejected (poison=True, correct=False)",
          r["poison"] and not r["correct"])
    check(f"poison: rejected instantly without judging (took {time.time()-t0:.2f}s)",
          (time.time() - t0) < 1.0)

    # Real judging path — only runs if the judger imports (sympy+antlr4 present).
    try:
        sys.path.insert(0, str(_REPO))
        import judger  # noqa: F401  (import canary)
        # Correct full-precision answer should PASS.
        r = judge_safe(r"the answer is \boxed{7091.66666666667}", ["7091.66666666667"], timeout=20)
        check("correct answer judged True", r["correct"] and not r["timed_out"])
        # Wrong answer should FAIL (not error).
        r = judge_safe(r"the answer is \boxed{3}", ["7091.66666666667"], timeout=20)
        check("wrong answer judged False", (not r["correct"]) and (not r["timed_out"]))
        # Representation equivalence (confirms we're on the NEW judger): 0.5 == 1/2.
        r = judge_safe(r"\boxed{0.5}", ["\\frac{1}{2}"], timeout=20)
        check("representation-equivalence 0.5 == 1/2 (new judge)", r["correct"])
    except Exception as e:
        print(f"  skip real-judge tests (judger import failed: {type(e).__name__}: {e})")
        print("  -> install sympy + antlr4-python3-runtime==4.11 to run them")

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_selftest())
