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

def reconstruct_prompt(user_question: str, retrieved_context: str, explainer_output: str, flag: str, context_from_prompt: str = ""):
    # conclusion, logical_evidence = "CONCLUSION", "LOGICAL EVIDENCE"
    # Use the provided explainer_output as the evidence
    conclusion = "See Logical Evidence"
    logical_evidence = explainer_output if explainer_output else "No logical evidence available."
    
    # Combine user-provided context (from prompt) with retrieved context
    full_context = ""
    if context_from_prompt:
        full_context += f"USER PROVIDED CONTEXT:\n{context_from_prompt}\n\n"
    if retrieved_context:
        full_context += f"RETRIEVED KNOWLEDGE:\n{retrieved_context}"
        
    context_part = f"CONTEXT: {full_context}" if flag != r"q" else ""
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
3. Multiple Choice/Categorical: If options are provided (like A, B, C, D) or if explicitly asked to answer with a specific format (e.g. True/False/Unknown), you MUST begin your answer with the selected option or category before any extra elaboration.
4. Never Cite Sources: Do NOT reference where you got the information from (no "according to...", "the context states...", "KBPedia defines...", etc.).
5. Smooth Tone: Write as a knowledgeable person would naturally explain something in conversation.
    """
     

if __name__ == "__main__":
    print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="q"), "\n")
    print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="x\c"), "\n")
    print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="x"), "\n")