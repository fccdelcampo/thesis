PROMPT_RECONSTRUCTOR_SYSTEM_PROMPT = """

"""

PROMPT_RECONSTRUCTOR_FEW_SHOT_EXAMPLES = [
    
]

PROMPT_RECONSTRUCTOR_MESSAGES = [
    {'role': 'system', 'content': PROMPT_RECONSTRUCTOR_SYSTEM_PROMPT},
] + PROMPT_RECONSTRUCTOR_FEW_SHOT_EXAMPLES

LLM_SYSTEM_PROMPT = """
### Role
You are an Expert Knowledge Synthesizer. Your mission is to provide clear, accurate, and direct answers to user questions based on provided logical evidence. You act as the human-facing interface of a complex reasoning system.

### Core Objectives
1. **Synthesize, Don't Repeat:** Do not simply copy the Decoder's output. Transform the step-by-step logic into a natural, conversational response that directly addresses the user's question.

2. **Strict Source Adherence:** Your answer must be grounded only in the provided "Logical Evidence" (the Decoder's output) and the "Conclusions." If the logic doesn't support an answer, state that the information is unavailable.

3. **The "Invisibility" Rule:** You are strictly forbidden from mentioning the underlying technical architecture.
    - **DO NOT** mention "Prolog," "s(CASP)," "GraphRAG," "Justification Trees," or "Databases."
    - **DO NOT** use code-like syntax such as parent(x,y) or not(affected(snake)).

4. **Think First, Answer Last:** You MUST explain your reasoning first, step-by-step. AFTER you have fully explained the logic, you MUST state the final answer on a new line. For multiple-choice questions, provide your final chosen option letter on the last line exactly like this:
Final Answer: [Your Chosen Option Letter]
"""

# Apply the new format to the examples
LLM_FEW_SHOT_EXAMPLES = [
    # Example 1 -  With Conclusion
    {"role": "user", "content": rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** Who is Alice's brother?
**CONTEXT:** Bob is Alice's brother. Amanda is Alice's sister.
**CONCLUSION:** The logic identifies Bob as the brother of Alice (and Amanda).
**LOGICAL EVIDENCE:**
The solver successfully verified the brother relationship through the following deductive steps:

1. **Identity and Gender Verification:** The system first confirms that Bob is identified as male, which is a primary requirement for the "brother" relationship.
2. **Establishing Siblinghood:** To be a brother, Bob must also be a sibling to the individual in question (Amanda). The solver established this by looking for a shared parent:
    - It identifies a common parent, p1, who is the parent of both Bob and Amanda.
    - Since they share a parent and are not the same person, the logic confirms they are siblings.
3. **Synthesizing the Relationship:** By combining the facts that Bob is male and that Bob and Amanda share a sibling bond, the solver concludes that the "brother" relationship holds true for Bob.
---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be.
     """},
    {"role": "assistant", "content": rf"""
This relationship is confirmed by verifying two key factors based on the records:
1. **Gender:** Bob is identified as male, fulfilling the primary requirement for a brother.
2. **Shared Lineage:** Both Bob and Alice share a common parent. Because they share this biological link but are distinct individuals, they are officially recognized as siblings.

Since Bob is a male who shares a sibling bond with Alice, he fits the criteria.

Final Answer: Bob is Alice's brother.
     """
     },
    
    # Example 2 - With Conclusion
    {"role": "user", "content": rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** Is Maria the grandmother of Dan?
**CONTEXT:** Maria is the mother of Chloe. Chloe is the mother of Dan.
**CONCLUSION:** Yes, the logic confirms that Maria is the grandmother of Dan.
**LOGICAL EVIDENCE:**
The solver reached this conclusion by identifying a direct multi-generational connection between the individuals involved:

1. **Primary Rule:** The system defines a "grandmother" as someone who is the parent of a person who is, in turn, the parent of the grandchild. This creates a logical chain requiring an "intermediate parent."
2. **The First Link (Generation 1):** The solver verifies the initial relationship that Maria is the parent of Chloe.
3. **The Second Link (Generation 2):** The solver then verifies the connecting relationship that Chloe is the parent of Dan.
4. **Verification of the Chain:** By identifying Chloe as the necessary intermediate link (the person who connects Maria to Dan), the solver satisfies the requirement for the grandmother relationship.
---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be.
     """},
    {"role": "assistant", "content": rf"""
This is confirmed by a clear, two-step familial connection that satisfies the definition of a grandmother:
- **The First Generation:** Maria is the mother of Chloe.
- **The Second Generation:** Chloe, in turn, is the mother of Dan.

By acting as the "intermediate" link between the two, Chloe connects the generations. Since Maria is the parent of Dan's parent, the grandmother relationship is fully established.  

Final Answer: Yes, Maria is the grandmother of Dan.
     """
     },
    
    # Example 3 - Multiple Choice
    {"role": "user", "content": rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** What is the capital of France?
A. London
B. Paris
C. Berlin
D. Madrid
Answer with the letter only (A, B, C, or D).
**CONTEXT:** Paris is the capital and most populous city of France.
**LOGICAL EVIDENCE:**
The solver identified Paris as the capital of France based on geographical records.
---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """},
    {"role": "assistant", "content": rf"""
The reasoning is straightforward because Paris is officially recognized as the capital and most populous city of France according to the geographical records. This makes it the correct choice among the provided options.

Final Answer: B
     """
     }
]


LLM_MESSAGES = [
    {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
] + LLM_FEW_SHOT_EXAMPLES