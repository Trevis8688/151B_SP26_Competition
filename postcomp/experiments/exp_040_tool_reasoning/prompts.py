"""Prompts for exp_040 tool-augmented reasoning (one-shot PAL).

Two free-form variants are defined so the DSMLP probe can A/B them empirically and
settle the advisor-vs-memory tension:
  * advisor: include a worked demo — instruction-only compliance is unreliable here.
  * memory (exp_005): Qwen3-Thinking regurgitates few-shot *answer numbers* verbatim
    under uncertainty. So the demo below teaches ONLY the FORMAT, using a neutral
    computation whose value won't plausibly transfer to a real question.

MCQ is unchanged from exp_015 (out of scope — tools don't help letter-selection and
add format risk). FEWSHOT_MCQ kept verbatim so MCQ stays in-distribution with GRPO
training; do not edit it.

Contract for one-shot PAL (parsed by postcomp/harness/pal.py):
  reason normally, then end with ONE ```python block that COMPUTES and PRINTS the
  final answer as:   ANSWER: <v1>, <v2>, ...   at FULL precision, never rounded.
"""

# Baseline free-form prompt (exp_015/exp_018 stage-1) — kept for apples-to-apples.
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

# ── (A) prompt-only: "compute exact, never round" — no tool execution ──────────
# Tests how much of the gap is pure display-rounding vs. imprecise hand-arithmetic.
SYSTEM_PROMPT_MATH_FULLPREC = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}.\n"
    "CRITICAL: Report every numeric answer to FULL precision — at least 10 significant "
    "figures, and NEVER round. If the problem says 'round to N decimal places', IGNORE that "
    "instruction and give the unrounded value (e.g. write 7091.666666667, not 7091.67; "
    "2.039716566, not 2.04). Exact fractions or symbolic forms (e.g. \\frac{85100}{12}, "
    "\\sqrt{3}) are also acceptable when exact."
)

# ── (B) one-shot PAL: reason, then a final code block computes the answer ───────
SYSTEM_PROMPT_MATH_PAL = (
    "You are an expert mathematician with a Python interpreter. Solve the problem "
    "step-by-step. For the final numerical computation, write a single Python code block "
    "that computes the answer EXACTLY and prints it at FULL precision. You may use sympy, "
    "numpy, math, and mpmath (high precision). End your code by printing a line of the form:\n"
    "    ANSWER: <value>\n"
    "For multiple sub-answers, print them comma-separated in problem order:\n"
    "    ANSWER: <v1>, <v2>, ...\n"
    "Use exact/symbolic computation where possible and NEVER round (ignore any "
    "'round to N decimal places' instruction — give the full-precision value). After the "
    "code, also put the final answer inside \\boxed{} as a fallback."
)

# Format-only demonstration. Teaches the code-block + ANSWER-line shape WITHOUT a
# transferable answer (neutral geometry value, unlikely to match a real question).
# Used only in the "with-demo" probe arm.
FEWSHOT_PAL_DEMO = [
    {
        "role": "user",
        "content": (
            "A rectangle has width 7 and height 3. What is the length of its diagonal? "
            "Round to two decimal places."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The diagonal of a rectangle is sqrt(width^2 + height^2). I will compute it "
            "exactly and report full precision (ignoring the rounding instruction, since the "
            "grader needs full precision).\n\n"
            "```python\n"
            "from sympy import sqrt, Integer, N\n"
            "d = sqrt(Integer(7)**2 + Integer(3)**2)   # exact: sqrt(58)\n"
            "print('ANSWER:', N(d, 15))\n"
            "```\n"
            "\\boxed{\\sqrt{58}}"
        ),
    },
]

# ── MCQ: unchanged from exp_015 (kept in-distribution with GRPO training) ───────
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

FEWSHOT_MATH: list = []

FEWSHOT_MCQ: list = [
    {
        "role": "user",
        "content": (
            "Find 1 over 6 + 1 over 8.\n\n"
            "Options:\n"
            "A. 7 over 24\nB. 2 over 14\nC. 1 over 4\nD. 7 over 48\nE. 2 over 24\n"
            "F. 1 over 14\nG. 1 over 2\nH. 8 over 14\nI. 0.21 Repeating\nJ. 4 over 24"
        ),
    },
    {
        "role": "assistant",
        "content": "Common denominator is 24: 1/6 = 4/24 and 1/8 = 3/24, so 4/24 + 3/24 = 7/24. \\boxed{A}",
    },
    {
        "role": "user",
        "content": (
            "The function value of $\\cos(\\pi + 5i)$ is ( ).\n\n"
            "Options:\n"
            "A. -cosh5\nB. -sinh5\nC. sin5i\nD. -sin5\nE. cos5\n"
            "F. cosh5i\nG. sinh5\nH. -cos5\nI. cosh5\nJ. -sin5i"
        ),
    },
    {
        "role": "assistant",
        "content": "$\\cos(\\pi + 5i) = -\\cos(5i) = -\\cosh(5)$, using $\\cos(\\pi+x) = -\\cos x$ and $\\cos(ix) = \\cosh x$. \\boxed{A}",
    },
    {
        "role": "user",
        "content": (
            "Find the range of $f(x) = \\frac{ x }{ 1+x^2 }$.\n\n"
            "Options:\n"
            "A. -\\frac{1}{3} \\le y \\le \\frac{1}{3}\n"
            "B. -\\frac{1}{\\sqrt{3}} \\le y \\le \\frac{1}{\\sqrt{3}}\n"
            "C. -\\frac{1}{4} \\le y \\le \\frac{1}{4}\n"
            "D. -\\frac{1}{\\sqrt{2}} \\le y \\le \\frac{1}{\\sqrt{2}}\n"
            "E. -\\frac{1}{\\sqrt{6}} \\le y \\le \\frac{1}{\\sqrt{6}}\n"
            "F. -\\frac{1}{2} \\le y \\le \\frac{1}{2}\n"
            "G. -\\frac{1}{\\sqrt{5}} \\le y \\le \\frac{1}{\\sqrt{5}}\n"
            "H. -1 \\le y \\le 1\n"
            "I. -\\frac{1}{\\sqrt{7}} \\le y \\le \\frac{1}{\\sqrt{7}}\n"
            "J. -\\frac{1}{\\sqrt{4}} \\le y \\le \\frac{1}{\\sqrt{4}}"
        ),
    },
    {
        "role": "assistant",
        "content": "$f'(x) = \\frac{1-x^2}{(1+x^2)^2} = 0$ at $x=\\pm 1$, giving $f(\\pm 1) = \\pm 1/2$. So range is $-1/2 \\le y \\le 1/2$. \\boxed{F}",
    },
]
