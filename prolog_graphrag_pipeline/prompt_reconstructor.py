import re
from typing import Optional

def capture_conclusion_and_logical_path(explainer_output):
    pattern = r"\*\*(?:Conclusion|CONCLUSION):\*\*\s*(.*)"
    match = re.search(pattern, explainer_output)
    if match:
        conclusion = match.group(1).strip()
        logical_evidence = explainer_output[match.end():].strip()
    else:
        conclusion = "Not extracted. Refer to logical evidence."
        logical_evidence = explainer_output
    return conclusion, logical_evidence

def reconstruct_prompt(user_question: str, retrieved_context: str, explainer_output: str, flag: str, context_from_prompt: str = "", verified_choice: Optional[str] = None, graphrag_answer: str = ""):
    conclusion = "See Logical Evidence"

    # Determine if Prolog generated verified evidence or if we are falling back
    prolog_verified = bool(explainer_output and explainer_output.strip() and
                           explainer_output.strip() != "No logical evidence available.")

    if prolog_verified:
        logical_evidence_label = "**PROLOG-VERIFIED LOGICAL EVIDENCE (Authoritative):**"
        logical_evidence_text = explainer_output
    elif graphrag_answer and graphrag_answer.strip():
        # Prolog failed — inject GraphRAG's preliminary analysis as the anchor
        logical_evidence_label = "**PRELIMINARY ANALYSIS (Knowledge Graph Retrieval):**"
        logical_evidence_text = graphrag_answer
    else:
        logical_evidence_label = "**LOGICAL EVIDENCE:**"
        logical_evidence_text = "No logical evidence available. Use your knowledge and the retrieved context to answer."

    # Combine user-provided context (from prompt) with retrieved context
    full_context = ""
    if context_from_prompt:
        full_context += f"USER PROVIDED CONTEXT:\n{context_from_prompt}\n\n"
    if retrieved_context:
        full_context += f"RETRIEVED KNOWLEDGE:\n{retrieved_context}"

    context_part = f"CONTEXT:\n{full_context}" if flag != r"q" else ""
    conclusion_part = f"CONCLUSION: {conclusion}" if flag == r"x" else ""
    logical_evidence_part = f"{logical_evidence_label}\n{logical_evidence_text}" if flag != r"q" else ""

    # Build authority instruction based on what evidence is available
    if prolog_verified and verified_choice:
        authority_instruction = f"""
╔══ CRITICAL ENFORCEMENT ══╗
The Prolog logic engine has formally PROVEN the answer is option [{verified_choice.upper()}].
You MUST output [{verified_choice.upper()}] as the answer letter.
DO NOT output any other letter. DO NOT say "None" or "None of the above".
Re-read the question choices: the text for [{verified_choice.upper()}] IS one of the listed options.
╚══════════════════════════╝"""
    elif prolog_verified:
        authority_instruction = """
╔══ CRITICAL ENFORCEMENT ══╗
The PROLOG-VERIFIED LOGICAL EVIDENCE above is the output of a formal logic engine.
You MUST base your answer letter DIRECTLY on this evidence.
DO NOT override it. DO NOT say "None" or "None of the above".
╚══════════════════════════╝"""
    elif graphrag_answer and graphrag_answer.strip():
        authority_instruction = """
╔══ MANDATORY INSTRUCTION ══╗
The PRELIMINARY ANALYSIS above contains a suggested answer from the knowledge graph.
You MUST treat this as your primary reference. Extract the answer letter it suggests and USE IT.
DO NOT ignore it. DO NOT say "None" or "None of the above" if the analysis points to a valid choice.
╚═══════════════════════════╝"""
    else:
        authority_instruction = ""

    return f"""### FINAL SYNTHESIS TASK

USER QUESTION:
{user_question}

{context_part}
{conclusion_part}
{logical_evidence_part}
{authority_instruction}

---
### OUTPUT FORMAT — MANDATORY:

Your response MUST begin with EXACTLY this phrase on the very first line:
  "Based on my synthesis of knowledge, the answer is [Letter]. [Exact choice text]."

Where [Letter] is ONE of: A, B, C, or D — matching EXACTLY the letter label in the question.
Where [Exact choice text] is the VERBATIM text of that choice from the question.

LETTER-TEXT ALIGNMENT: The letter and text MUST match the same choice in the question. If you write "B.", the text after it MUST be what choice B says — not choice A or C.
NEVER write "None", "None of the above", or invent a choice not listed in the question.
After the first line, you may add 1-2 sentences of explanation.

Example of correct format:
Based on my synthesis of knowledge, the answer is B. soda can.
    """


if __name__ == "__main__":
    print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="q"), "\n")
    print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag=r"x\c"), "\n")
    print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="x"), "\n")