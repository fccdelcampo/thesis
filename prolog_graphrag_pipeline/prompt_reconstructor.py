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
    conclusion, logical_evidence = "CONCLUSION", "LOGICAL EVIDENCE"
    # conclusion, logical_evidence = capture_conclusion_and_logical_path(explainer_output=explainer_output)
    context_part = f"**CONTEXT:** {retrieved_context}" if flag is not r"q" else ""
    conclusion_part = f"**CONCLUSION:** {conclusion}" if flag==r"x" else ""
    logical_evidence_part = f"**LOGICAL EVIDENCE:** {logical_evidence}" if flag is not r"q" else ""
    return rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** {user_question}
{context_part}
{conclusion_part}
{logical_evidence_part}

---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """
     

print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="q"), "\n")
print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="x\c"), "\n")
print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="x"), "\n")