import re
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

def reconstruct_prompt(user_question: str, retrieved_context: str, explainer_output: str, flag: str):
    # conclusion, logical_evidence = "CONCLUSION", "LOGICAL EVIDENCE"
    # Use the provided explainer_output as the evidence
    conclusion = "See Logical Evidence"
    logical_evidence = explainer_output if explainer_output else "No logical evidence available."
    context_part = f"CONTEXT: {retrieved_context}" if flag != r"q" else ""
    conclusion_part = f"CONCLUSION: {conclusion}" if flag == r"x" else ""
    logical_evidence_part = f"**LOGICAL EVIDENCE: {logical_evidence}" if flag != r"q" else ""
    return f"""
### FINAL SYNTHESIS TASK

USER QUESTION: {user_question}
{context_part}
{conclusion_part}
{logical_evidence_part}

---
### INSTRUCTION:
Provide a direct, confident answer to the User Question.
1. Answer Naturally: Respond as if you already know the answer. Do NOT mention "the context", "the provided information", or "based on the evidence".
2. Be Direct: Lead with the answer, then elaborate if helpful.
3. Never Cite Sources: Do NOT reference where you got the information from (no "according to...", "the context states...", "YAGO defines...", etc.).
4. Smooth Tone: Write as a knowledgeable person would naturally explain something in conversation.
    """
     

print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="q"), "\n")
print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="x\c"), "\n")
print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="x"), "\n")