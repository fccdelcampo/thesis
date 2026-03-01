import ollama

LLM_MESSAGES = [
    # System Prompt
    {"role": "system",
     "content": """
### ROLE:
You are an expert Logical Analyst specialized in symbolic reasoning and Natural Language Inference (NLI).

### TASK:
Determine if two LLM responses, generated from a Prolog-GraphRAG pipeline, are "Semantically Equivalent." 

### DEFINITION OF EQUIVALENCE:
Two responses are equivalent ONLY if:
1. They reach the identical final conclusion (e.g., the specific answer to the question).
2. They utilize the same core logical triples (retrieved facts) and Prolog rules to support that conclusion.
3. Differences in stylistic phrasing, explanation length, or the order of sentences do NOT constitute a difference in meaning.

### REJECTION CRITERIA:
You should reject equivalence (verdict: "no") if ANY of the following apply:
- One response contains a logical contradiction to the other.
- One response includes a pivotal fact or deduction that changes the validity of the conclusion.
- The final answers are different (e.g., True vs. Unknown).

### JSON SCHEMA:
{
  "rationale": "A concise explanation of why the two responses are semantically equivalent or not.",
  "verdict": "yes" | "no"
}     
     """}
]


def generate(prompt: str):
    try:
        response = ollama.chat(
                model='llama3:instruct',
                messages=LLM_MESSAGES + [{"role": "user", "content": prompt}],
                options={
                    'temperature': 0.3,
                },
                logprobs=True     
            )
        answer = response['message']['content']
        return answer
    except Exception as e:
        print(f"Error during Ollama interaction: {e}")
        return None
        