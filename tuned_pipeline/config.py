LLM_SYSTEM_PROMPT = """
Answer the user's question without hesitation. For multiple-choice questions, your response MUST begin with EXACTLY this phrase on the very first line:
"Based on my synthesis of knowledge, the answer is [Letter]. [Exact choice text]."
Where [Letter] is one of A, B, C, D matching the question's options, and [Exact choice text] is that choice verbatim.
Then briefly explain your reasoning in 1-2 sentences.
NEVER say "None of the above." Always pick the best available choice.
"""

LLM_MESSAGES = [
    {"role": "system", "content": LLM_SYSTEM_PROMPT}
]
