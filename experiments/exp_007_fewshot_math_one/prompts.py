"""System prompts for this experiment. Copied into the notebook at run time.

Change these, not the notebook, so every experiment's prompts are captured in git.
"""

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

# One free-form example: geometric series with a symbolic fractional answer.
# Symbolic answer (not an integer) prevents regurgitation.
# No fake <think> tags — the model generates its own CoT.
FEWSHOT_MATH: list = [
    {
        "role": "user",
        "content": (
            "Find the sum of the infinite geometric series "
            "$\\sum_{n=0}^{\\infty} \\frac{3}{4^n}$."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "This is an infinite geometric series with first term $a = 3$ and common ratio $r = \\frac{1}{4}$. "
            "Since $|r| < 1$, the series converges. "
            "The sum is $S = \\frac{a}{1 - r} = \\frac{3}{1 - \\frac{1}{4}} = \\frac{3}{\\frac{3}{4}} = 4$. "
            "\\boxed{4}"
        ),
    },
]

# MCQ examples — unchanged from exp_004 (proven at 63.2%).
FEWSHOT_MCQ: list = [
    {
        "role": "user",
        "content": (
            "Find 1 over 6 + 1 over 8.\n\n"
            "Options:\n"
            "A. 7 over 24\n"
            "B. 2 over 14\n"
            "C. 1 over 4\n"
            "D. 7 over 48\n"
            "E. 2 over 24\n"
            "F. 1 over 14\n"
            "G. 1 over 2\n"
            "H. 8 over 14\n"
            "I. 0.21 Repeating\n"
            "J. 4 over 24"
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
            "A. -cosh5\n"
            "B. -sinh5\n"
            "C. sin5i\n"
            "D. -sin5\n"
            "E. cos5\n"
            "F. cosh5i\n"
            "G. sinh5\n"
            "H. -cos5\n"
            "I. cosh5\n"
            "J. -sin5i"
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
