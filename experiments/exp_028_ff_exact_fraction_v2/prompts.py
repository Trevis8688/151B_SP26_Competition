"""System prompts for this experiment. Copied into the notebook at run time.

Change these, not the notebook, so every experiment's prompts are captured in git.

IMPORTANT: These prompts MUST match the prompts used during GRPO training
(Cell 5 of train_grpo.ipynb). The trained model expects to see the 3 MCQ
few-shot examples in its context — running inference without them puts the
model out-of-distribution from training, which will degrade MCQ accuracy.

exp_028 (v2) single change vs exp_027: the free-form precision instruction now
PROTECTS closed-form symbolic constants (pi, e, sqrt, surds) — keep them symbolic
rather than converting to a decimal. This plugs the exp_027 leak (id=429: model
turned symbolic e^2 into 7.389056099, gold wanted e^2, scored wrong). MCQ prompt +
few-shots remain byte-identical to exp_024. Hypothesis unchanged: recover the
wrong_math PRECISION bucket (right values judged wrong at the judger's 1e-08
tolerance from 2-4-digit rounding) via exact fractions / >=10 sig figs, WITHOUT
introducing a symbolic->decimal regression. Dev-only probe; watch MCQ for noise.
"""

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}. "
    "Report exact values. If an answer is rational, write it as an exact fraction "
    "(for example \\boxed{a/b}) instead of a decimal. If an answer is a closed-form expression "
    "involving constants such as \\pi, e, or square roots, keep it in that exact symbolic form "
    "(for example \\boxed{e^2} or \\boxed{3\\sqrt{2}}) — do not convert it to a decimal. "
    "Only when an answer can be given solely as a decimal, write at least 10 significant figures "
    "and do not round to fewer digits."
)

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
