"""System prompts for this experiment. Copied into the notebook at run time.

Change these, not the notebook, so every experiment's prompts are captured in git.
"""

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Use \\boxed{} ONLY for the final answer(s) — never for intermediate work or scratchwork. "
    "If the problem asks for N values (e.g., quartiles, multiple coefficients, list of roots), "
    "produce exactly N separate \\boxed{} calls in the order requested, e.g. \\boxed{Q1} \\boxed{median} \\boxed{Q3}. "
    "If the problem asks for one value, produce one \\boxed{} at the end. "
    "Never emit an empty \\boxed{}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

# Few-shot examples — list of {"role": "user"/"assistant", "content": ...}
# inserted between the system prompt and the actual question.
#
# Math examples teach the format discipline that exp_004 analysis showed was
# the dominant free-form failure mode:
#   - one \boxed{} per gold value, with the correct count
#   - no \boxed{} on intermediate work
#   - finish with the boxed values

FEWSHOT_MATH: list = [
    {
        "role": "user",
        "content": (
            "Find the first quartile $Q_1$, the median, and the third quartile $Q_3$ "
            "of the data set: 4, 7, 8, 11, 13, 16, 18, 21."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "There are 8 sorted values, so the median is the average of the 4th and 5th: "
            "(11+13)/2 = 12. The lower half is 4, 7, 8, 11; its median (Q1) is (7+8)/2 = 7.5. "
            "The upper half is 13, 16, 18, 21; its median (Q3) is (16+18)/2 = 17. "
            "Three values are requested, so I emit three separate boxes in the order Q1, median, Q3.\n\n"
            "\\boxed{7.5} \\boxed{12} \\boxed{17}"
        ),
    },
    {
        "role": "user",
        "content": "Evaluate $\\int_0^{\\pi/2} \\sin x \\cos x \\, dx$.",
    },
    {
        "role": "assistant",
        "content": (
            "Let $u = \\sin x$, so $du = \\cos x \\, dx$. When $x=0$, $u=0$; when $x=\\pi/2$, $u=1$. "
            "The integral becomes $\\int_0^1 u \\, du = u^2/2 \\Big|_0^1 = 1/2$. "
            "One value requested, so one box.\n\n"
            "\\boxed{\\frac{1}{2}}"
        ),
    },
    {
        "role": "user",
        "content": (
            "A line passes through the points $(1, 5)$ and $(3, 11)$. "
            "Write its equation in the form $y = mx + b$, and then find $y$ when $x = 7$."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Slope: $m = (11-5)/(3-1) = 3$. Intercept: $5 = 3(1) + b \\Rightarrow b = 2$, "
            "so the equation is $y = 3x + 2$. At $x = 7$: $y = 3(7)+2 = 23$. "
            "Two values requested (the equation and the y-value), so two separate boxes.\n\n"
            "\\boxed{y = 3x + 2} \\boxed{23}"
        ),
    },
]

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
