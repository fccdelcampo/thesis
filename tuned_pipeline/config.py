LLM_SYSTEM_PROMPT = """
Simply answer the user's question without hesitation. For multiple-choice questions, respond EXACTLY with:
"The answer is [Letter]. [Exact choice text]."
Where [Letter] is one of A, B, C, D matching the question's options, and [Exact choice text] is that choice verbatim.
Then rationalize your answer.
"""

LLM_MESSAGES = [
    {"role": "system", "content": LLM_SYSTEM_PROMPT}
]
