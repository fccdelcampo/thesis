LLM_SYSTEM_PROMPT = """
Answer the user's question. If it is a multiple choice question, explain very briefly first then answer with your choice.
"""

LLM_MESSAGES = [
    {"role": "system", "content": LLM_SYSTEM_PROMPT}
]