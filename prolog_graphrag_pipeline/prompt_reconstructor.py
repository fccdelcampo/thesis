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
Provide a final, coherent response to the User Question.
1. Synthesize: Use the Logical Evidence and Context to explain the answer.
2. Do Not List: Do NOT simply list the retrieved facts or YAGO definitions. Integrate them into paragraphs.
3. Tone: Ensure the tone is natural and free of technical jargon.
     """
     

print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="q"), "\n")
print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="x\c"), "\n")
print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="x"), "\n")