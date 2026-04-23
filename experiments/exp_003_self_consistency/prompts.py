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

# Optional: few-shot examples. List of {"role": "user"/"assistant", "content": ...}
# inserted between the system prompt and the actual question.
FEWSHOT_MATH: list = []
FEWSHOT_MCQ: list = []
