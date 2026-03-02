import sys
import os

# Ensure project root is on the path so llm_config can be imported
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from llm_config import MODEL_NAME, get_openai_client

client = get_openai_client()

# ── System Prompt ────────────────────────────────────────────────────────────
LLM_MESSAGES = [
    {
        "role": "system",
        "content": """\
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
""",
    }
]


# ── Unified Generation Function ─────────────────────────────────────────────
def generate(prompt: str):
    """Send a prompt to the configured LLM and return the response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=LLM_MESSAGES + [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during LLM interaction ({MODEL_NAME}): {e}")
        return None