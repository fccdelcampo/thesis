LLM_SYSTEM_PROMPT = """
Answer the question given to you.
"The answer is [Letter]. [Exact choice text]."
Where [Letter] is one of A, B, C, D matching the question's options, and [Exact choice text] is that choice verbatim.
"""

LLM_MESSAGES = [
    {"role": "system", "content": LLM_SYSTEM_PROMPT}
]
