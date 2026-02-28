LLM_SYSTEM_PROMPT = """
Answer the user's question. If it is a multiple choice question, answer with the letter of the correct choice first before explaining.
"""

LLM_MESSAGES = [
    {"role": "system", "content": LLM_SYSTEM_PROMPT}
]