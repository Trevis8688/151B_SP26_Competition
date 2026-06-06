#!/usr/bin/env python3
"""One-shot PAL (Program-Aided) answer assembly for exp_040.

The contract with the model (set up by the prompt — see prompts.py):

  * Reason normally.
  * End with ONE ```python``` block that computes the answer and prints it on a
    final line as:   ANSWER: <v1>, <v2>, ...
    using FULL precision (no rounding) — for multi-part questions, all values
    comma-separated in problem order.

This module is the GPU-free glue: it pulls that final code block, runs it through
the sandbox (executor.py), and assembles a judge-ready response whose `\boxed{}`
holds the tool's full-precision output. If the model emitted no runnable block, or
the code failed/timed out, we fall back to the model's own `\boxed{}` so PAL can
never score *below* the plain baseline on a given item.

Why this shape (see postcomp/DEVLOG.md):
  * The judge demands ~8 sig figs; the model reasons correctly but rounds. Taking
    the printed full-precision value fixes exactly that failure.
  * Multi-value answers must be emitted correctly from the start (precision and
    multi-part errors are entangled — a single-value-only PAL would address < 1/3
    of the recoverable errors).

No GPU and no model here. Run `python postcomp/harness/pal.py` for self-tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from .executor import ExecResult, run_code
except ImportError:  # allow running as a script
    from executor import ExecResult, run_code

# Matches ```python ... ``` (or bare ``` ... ```) fenced blocks, non-greedy.
_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANSWER_LINE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_BOXED = re.compile(r"\\boxed\{")


def extract_code_blocks(text: str) -> list[str]:
    """All fenced code blocks, in order."""
    return [m.group(1).strip() for m in _CODE_FENCE.finditer(text or "")]


def last_code_block(text: str) -> str | None:
    """The final fenced code block — PAL's 'compute the answer' block."""
    blocks = extract_code_blocks(text)
    return blocks[-1] if blocks else None


def extract_boxed(text: str) -> str | None:
    """Content of the last `\\boxed{...}` (brace-balanced). None if absent."""
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
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


def parse_tool_answer(stdout: str) -> str | None:
    """Pull the answer string from sandbox stdout.

    Prefers the last `ANSWER: ...` line (the contract); otherwise falls back to the
    last non-empty line, so a model that just `print(...)`s still works.
    """
    if not stdout:
        return None
    matches = _ANSWER_LINE.findall(stdout)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else None


@dataclass
class PalOutcome:
    response: str          # judge-ready text (contains the chosen \boxed{})
    used_tool: bool        # True iff the tool answer was used for the box
    tool_answer: str | None
    exec_ok: bool
    reason: str            # why we did/didn't use the tool (for diagnostics)


def assemble(
    model_text: str,
    exec_result: ExecResult | None,
) -> PalOutcome:
    """Combine the model's text with a sandbox result into a final response.

    Decision order:
      1. Tool ran, produced a usable ANSWER → box the tool's full-precision answer.
      2. Otherwise → keep the model's own response (its \boxed{} stands as fallback).

    The returned `response` always contains a `\boxed{}` if either source had one,
    so it scores through the unchanged judger path.
    """
    model_box = extract_boxed(model_text)

    if exec_result is not None and exec_result.ok:
        tool_ans = parse_tool_answer(exec_result.stdout)
        if tool_ans:
            # Preserve the model's reasoning for transparency, then override the box.
            response = (
                model_text.rstrip()
                + "\n\n[TOOL EXECUTION]\n"
                + f"ANSWER: {tool_ans}\n"
                + f"\\boxed{{{tool_ans}}}"
            )
            return PalOutcome(response, True, tool_ans, True, "used tool answer")
        reason = "tool ran but no parseable ANSWER line"
    elif exec_result is None:
        reason = "no code block to execute"
    elif exec_result.timed_out:
        reason = "tool timed out"
    else:
        reason = f"tool error (rc={exec_result.returncode})"

    # Fallback: model's own response unchanged.
    return PalOutcome(model_text, False, None, False, reason + " -> fallback to model box")


def run_pal(model_text: str, timeout_s: float = 5.0, mem_mb: int = 4096) -> PalOutcome:
    """End-to-end: extract last code block, execute it, assemble. (Touches no GPU.)"""
    code = last_code_block(model_text)
    exec_result = run_code(code, timeout_s=timeout_s, mem_mb=mem_mb) if code else None
    return assemble(model_text, exec_result)


# ─────────────────────────────────────────────────────────────────────────────
# Self-tests — run `python postcomp/harness/pal.py`  (uses a mock model; no GPU)
# ─────────────────────────────────────────────────────────────────────────────
def _selftest() -> int:
    failures = []

    def check(name, cond):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            failures.append(name)

    print("pal self-tests:")

    # --- pure parsing (no subprocess) ---
    txt = "reasoning...\n```python\nprint('ANSWER: 42')\n```\nmore\n```python\nx=1\n```"
    check("last_code_block picks the final block", last_code_block(txt) == "x=1")
    check("extract_boxed balanced", extract_boxed(r"foo \boxed{\frac{1}{2}} bar") == r"\frac{1}{2}")
    check("extract_boxed none", extract_boxed("no box here") is None)
    check("parse_tool_answer ANSWER line", parse_tool_answer("noise\nANSWER: 2.5, 7\n") == "2.5, 7")
    check("parse_tool_answer fallback last line", parse_tool_answer("3.14159") == "3.14159")

    # --- assemble: fallback when no exec ---
    out = assemble(r"text \boxed{7.0}", None)
    check("assemble no-exec falls back", (not out.used_tool) and extract_boxed(out.response) == "7.0")

    # --- assemble: tool error falls back to model box ---
    bad = ExecResult(ok=False, stdout="", stderr="ZeroDivisionError", timed_out=False, returncode=1)
    out = assemble(r"text \boxed{5}", bad)
    check("assemble tool-error falls back", (not out.used_tool) and extract_boxed(out.response) == "5")

    # --- assemble: tool success overrides the box ---
    good = ExecResult(ok=True, stdout="ANSWER: 7091.66666666667\n", stderr="", timed_out=False, returncode=0)
    out = assemble(r"text \boxed{7091.67}", good)
    check("assemble tool-success overrides box",
          out.used_tool and extract_boxed(out.response) == "7091.66666666667")

    # --- end-to-end through the real sandbox (the motivating case) ---
    model = (
        "She earns 85100 over 12 months.\n"
        "```python\n"
        "from sympy import Rational\n"
        "print(f'ANSWER: {float(Rational(85100,12))}')\n"
        "```\n"
        r"So the monthly salary is \boxed{7091.67}."  # model rounded — should be overridden
    )
    out = run_pal(model)
    box = extract_boxed(out.response)
    check("e2e: tool full-precision overrides rounded box",
          out.used_tool and box is not None and box.startswith("7091.666666"))

    # --- e2e: bad code falls back, never crashes ---
    out = run_pal("text\n```python\nthis is not valid python(((\n```\n" + r"\boxed{3}")
    check("e2e: invalid code falls back to model box", (not out.used_tool) and extract_boxed(out.response) == "3")

    # --- judge integration: the overridden answer actually passes the real judger ---
    try:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
        from judger import Judger
        J = Judger(strict_extract=False)
        passed = bool(J.auto_judge(pred=out_box_resp(), gold=["7091.66666666667"], options=[[]]))
        check("judge: PAL full-precision answer PASSES real judger", passed)
    except Exception as e:
        print(f"  skip judge-integration test ({type(e).__name__}: {e})")

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


def out_box_resp() -> str:
    """Helper for the judge-integration self-test: a PAL response for the salary case."""
    good = ExecResult(ok=True, stdout="ANSWER: 7091.66666666667\n", stderr="", timed_out=False, returncode=0)
    return assemble(r"text \boxed{7091.67}", good).response


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
