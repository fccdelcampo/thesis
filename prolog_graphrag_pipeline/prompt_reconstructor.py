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
    conclusion = "See Logical Evidence"
    
    # Determine if Prolog generated verified evidence or if we are falling back
    prolog_verified = bool(explainer_output and explainer_output.strip() and
                           explainer_output.strip() != "No logical evidence available.")
    
    if prolog_verified:
        logical_evidence_label = "**PROLOG-VERIFIED LOGICAL EVIDENCE (Authoritative — follow this conclusion):"
        logical_evidence_text = explainer_output
    else:
        logical_evidence_label = "**LOGICAL EVIDENCE:"
        logical_evidence_text = "No logical evidence available. Use your knowledge and the retrieved context to answer."

    # Combine user-provided context (from prompt) with retrieved context
    full_context = ""
    if context_from_prompt:
        full_context += f"USER PROVIDED CONTEXT:\n{context_from_prompt}\n\n"
    if retrieved_context:
        full_context += f"RETRIEVED KNOWLEDGE:\n{retrieved_context}"
        
    context_part = f"CONTEXT: {full_context}" if flag != r"q" else ""
    conclusion_part = f"CONCLUSION: {conclusion}" if flag == r"x" else ""
    logical_evidence_part = f"{logical_evidence_label} {logical_evidence_text}" if flag != r"q" else ""
    
    authority_instruction = ""
    if prolog_verified:
        authority_instruction = """
CRITICAL: The PROLOG-VERIFIED LOGICAL EVIDENCE above is the output of a formal logic engine that has already proven the correct answer. You MUST state a Final Answer that matches this evidence. Do NOT override it with your own intuition or background knowledge."""
    
    return f"""
### FINAL SYNTHESIS TASK

USER QUESTION: {user_question}
{context_part}
{conclusion_part}
{logical_evidence_part}
{authority_instruction}

---
### INSTRUCTION:
Provide a direct, confident answer to the User Question.
1. Answer Naturally: Respond as if you already know the answer. Do NOT mention "the context", "the provided information", or "based on the evidence".
2. Be Direct: Lead with the Final Answer letter (for MCQ), then elaborate briefly.
3. Multiple Choice/Categorical: If options are provided (like A, B, C, D), you MUST begin your answer with:
Final Answer: [Letter]
4. Never Cite Sources: Do NOT reference where you got the information from.
5. Smooth Tone: Write as a knowledgeable person would naturally explain something in conversation.
    """
     

if __name__ == "__main__":
    print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="q"), "\n")
    print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="x\c"), "\n")
    print(reconstruct_prompt(user_question="QUESTION", retrieved_context="CONTEXT", explainer_output="", flag="x"), "\n")