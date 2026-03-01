LLM_SYSTEM_PROMPT = """
Answer the user's question. If it is a multiple choice question, strictly answer with Final Answer: [LETTER] then briefly explain.
"""

LLM_MESSAGES = [
    {"role": "system", "content": LLM_SYSTEM_PROMPT}
]