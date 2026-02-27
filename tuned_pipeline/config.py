LLM_SYSTEM_PROMPT = """
Answer the user's question. If it is a multiple choice question, answer with the letter of the correct option and nothing else.
"""

LLM_MESSAGES = [
    {"role": "system", "content": LLM_SYSTEM_PROMPT}
]