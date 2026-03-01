LLM_SYSTEM_PROMPT = """
Answer the user's question without hesitation. If it is a multiple choice question, answer with exactly this format:
*[Letter].* [Choice text]
Then briefly explain your reasoning.
"""

LLM_MESSAGES = [
    {"role": "system", "content": LLM_SYSTEM_PROMPT}
]