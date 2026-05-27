"""exp_036 verification-probe prompts.

The verifier sees the question + the model's PROPOSED answer (not the prior
chain of thought) and is asked to independently re-derive and judge.

Two prompt variants for the probe to A/B which works on Qwen3-Thinking:
  - SYSTEM_PROMPT_VERIFY_FF   : free-form questions
  - SYSTEM_PROMPT_VERIFY_MCQ  : MCQ questions

The fewshots are deliberately small (1 example each) — too many fewshots cause
Qwen3-Thinking to mode-lock to the verdict pattern (exp_005 lesson).
"""

# ---------------------------------------------------------------------------
# Free-form verifier
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_VERIFY_FF = (
    "You are verifying whether a proposed answer to a math problem is correct.\n"
    "You did NOT solve this problem before; treat the proposed answer as a hypothesis.\n"
    "Carefully redo the key computation. Be honest — if the proposed answer is wrong,\n"
    "say so even if it looks plausible.\n\n"
    "Your final output MUST contain a single line of the form:\n"
    "  VERDICT: \\boxed{CORRECT}\n"
    "OR\n"
    "  VERDICT: \\boxed{INCORRECT}\n\n"
    "If you say INCORRECT and you have a confident corrected answer, append it on a\n"
    "new line as:\n"
    "  CORRECTED: \\boxed{<your answer>}\n"
    "If you are unsure of the correct answer, omit the CORRECTED line."
)

# ---------------------------------------------------------------------------
# MCQ verifier (kept here for completeness; primary probe is FF)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_VERIFY_MCQ = (
    "You are verifying whether a proposed answer letter to a multiple-choice math\n"
    "problem is correct. You did NOT pick this letter before; treat it as a hypothesis.\n"
    "Carefully redo the key computation, then compare to the options.\n\n"
    "Your final output MUST contain a single line of the form:\n"
    "  VERDICT: \\boxed{CORRECT}\n"
    "OR\n"
    "  VERDICT: \\boxed{INCORRECT}\n\n"
    "If you say INCORRECT, append the corrected letter on a new line as:\n"
    "  CORRECTED: \\boxed{<letter>}"
)

# ---------------------------------------------------------------------------
# Few-shot anchors (small — Qwen3-Thinking mode-locks on heavier fewshots).
# ---------------------------------------------------------------------------
FEWSHOT_VERIFY_FF = [
    {
        "role": "user",
        "content": (
            "Problem:\n"
            "What is the area of a triangle with base 8 and height 5?\n\n"
            "Proposed answer: 20"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "<think>Area = (1/2) * base * height = (1/2) * 8 * 5 = 20.</think>\n\n"
            "The proposed answer 20 matches the computation.\n\n"
            "VERDICT: \\boxed{CORRECT}"
        ),
    },
    {
        "role": "user",
        "content": (
            "Problem:\n"
            "Compute 12 * 13.\n\n"
            "Proposed answer: 154"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "<think>12 * 13 = 12 * (10 + 3) = 120 + 36 = 156, not 154.</think>\n\n"
            "The proposed answer 154 is off by 2. Correct value is 156.\n\n"
            "VERDICT: \\boxed{INCORRECT}\n"
            "CORRECTED: \\boxed{156}"
        ),
    },
]

FEWSHOT_VERIFY_MCQ = [
    {
        "role": "user",
        "content": (
            "Problem:\n"
            "Which of the following is a prime number?\n"
            "A. 4\nB. 9\nC. 7\nD. 15\n\n"
            "Proposed answer: B"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "<think>4=2*2, 9=3*3, 7 is prime (only divisors 1 and 7), 15=3*5.\n"
            "So C is prime, not B.</think>\n\n"
            "B (which is 9) is not prime — 9 = 3 * 3. The prime option is C (which is 7).\n\n"
            "VERDICT: \\boxed{INCORRECT}\n"
            "CORRECTED: \\boxed{C}"
        ),
    },
]


def build_verify_messages(question_text: str, proposed_answer: str, is_mcq: bool):
    """Construct the chat messages for a single verification call."""
    if is_mcq:
        system = SYSTEM_PROMPT_VERIFY_MCQ
        fewshot = FEWSHOT_VERIFY_MCQ
    else:
        system = SYSTEM_PROMPT_VERIFY_FF
        fewshot = FEWSHOT_VERIFY_FF
    user_msg = f"Problem:\n{question_text}\n\nProposed answer: {proposed_answer}"
    return [{"role": "system", "content": system}, *fewshot, {"role": "user", "content": user_msg}]
