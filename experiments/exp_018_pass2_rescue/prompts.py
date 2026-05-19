"""Stage 2 'boxed rescue' prompts — used only to extract a \\boxed{} answer
from a truncated stage-1 response.

These do NOT replace exp_009/exp_011 prompts. They're applied only to responses
that already failed to emit \\boxed{} during stage 1.
"""

RESCUE_SYSTEM_PROMPT_MATH = (
    "You are an answer extractor. The user provides a math problem and a partial "
    "reasoning trace that did not finish with a final answer. Based on the reasoning, "
    "output ONLY the final answer in \\boxed{}. If the problem has multiple sub-answers, "
    "separate them by commas inside a single \\boxed{}, e.g. \\boxed{3, 7}. "
    "Do not show your work. Do not restate the problem. Output only the box."
)

RESCUE_SYSTEM_PROMPT_MCQ = (
    "You are an answer extractor. The user provides a multiple-choice math problem "
    "and a partial reasoning trace that did not finish with a final answer. Based on "
    "the reasoning, output ONLY the chosen letter inside \\boxed{}, e.g. \\boxed{C}. "
    "Do not show your work. Do not restate the problem. Output only the box."
)


def build_rescue_user_message(question: str, options: list | None, partial_response: str) -> str:
    """Build the stage-2 user message from the original question + truncated stage-1 response."""
    parts = [f"PROBLEM:\n{question}"]
    if options:
        parts.append("OPTIONS:\n" + "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)))
    parts.append(f"PARTIAL REASONING (incomplete):\n{partial_response}")
    parts.append(
        "Based on the reasoning above, output ONLY your final answer in \\boxed{}. "
        "Do not show your work."
    )
    return "\n\n".join(parts)
