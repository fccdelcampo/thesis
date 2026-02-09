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

def reconstruct_prompt(user_question: str, retrieved_context: str, explainer_output: str, include_conclusion: bool = True):
    conclusion, logical_evidence = capture_conclusion_and_logical_path(explainer_output=explainer_output)
    conclusion_part = f"**CONCLUSION:** {conclusion}" if include_conclusion else ""
    return rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** {user_question}
**CONTEXT:** {retrieved_context}
{conclusion_part}
**LOGICAL EVIDENCE:** {logical_evidence}

---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """