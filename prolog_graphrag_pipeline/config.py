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

4. **Tone & Style:** Professional, helpful, and grounded. Use a "bottom-line up front" (BLUF) approach: answer the question immediately, then provide the supporting context.


"""

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
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """},
    {"role": "assistant", "content": rf"""
Based on the records, **Bob is Alice's brother**.

This relationship is confirmed by verifying two key factors:

    - **Gender:** Bob is identified as male, fulfilling the primary requirement for a brother.

    - **Shared Lineage:** Both Bob and Alice share a common parent. Because they share this biological link but are distinct individuals, they are officially recognized as siblings.

Since Bob is a male who shares a sibling bond with Alice, he is identified as her brother.
     """
     },
    
    # Example 1 - Without Conclusion
    {"role": "user", "content": rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** Who is Alice's brother?
**CONTEXT:** Bob is Alice's brother. Amanda is Alice's sister.
**LOGICAL EVIDENCE:**
The solver successfully verified the brother relationship through the following deductive steps:

1. **Identity and Gender Verification:** The system first confirms that Bob is identified as male, which is a primary requirement for the "brother" relationship.
2. **Establishing Siblinghood:** To be a brother, Bob must also be a sibling to the individual in question (Amanda). The solver established this by looking for a shared parent:
    - It identifies a common parent, p1, who is the parent of both Bob and Amanda.
    - Since they share a parent and are not the same person, the logic confirms they are siblings.
3. **Synthesizing the Relationship:** By combining the facts that Bob is male and that Bob and Amanda share a sibling bond, the solver concludes that the "brother" relationship holds true for Bob.

---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """},
    {"role": "assistant", "content": rf"""
**Bob is Alice's brother.**

This relationship is confirmed through a clear logical path:

    - **Gender Identification:** First, it is established that Bob is male, which is the fundamental requirement for the "brother" designation.

    - **Sibling Connection:** The records indicate that Bob and Alice share a common parent. In a family structure, sharing a parent makes individuals siblings.

By combining the fact that Bob is male with the confirmed sibling bond shared through a common parent, we can conclude that he is Alice's brother.
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
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """},
    {"role": "assistant", "content": rf"""
**Yes, Maria is the grandmother of Dan.**

This is confirmed by a clear, two-step familial connection that satisfies the definition of a grandmother:

    - **The First Generation:** Maria is the mother of Chloe.

    - **The Second Generation:** Chloe, in turn, is the mother of Dan.

By acting as the "intermediate" link between the two, Chloe connects the generations. Since Maria is the parent of Dan's parent, the grandmother relationship is fully established.  
     """
     },
    
    # Example 2 - Without Conclusion
    {"role": "user", "content": rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** Is Maria the grandmother of Dan?
**CONTEXT:** Maria is the mother of Chloe. Chloe is the mother of Dan.
**LOGICAL EVIDENCE:**
The solver reached this conclusion by identifying a direct multi-generational connection between the individuals involved:

1. **Primary Rule:** The system defines a "grandmother" as someone who is the parent of a person who is, in turn, the parent of the grandchild. This creates a logical chain requiring an "intermediate parent."
2. **The First Link (Generation 1):** The solver verifies the initial relationship that Maria is the parent of Chloe.
3. **The Second Link (Generation 2):** The solver then verifies the connecting relationship that Chloe is the parent of Dan.
4. **Verification of the Chain:** By identifying Chloe as the necessary intermediate link (the person who connects Maria to Dan), the solver satisfies the requirement for the grandmother relationship.
---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """},
    {"role": "assistant", "content": rf"""
**Yes, Maria is the grandmother of Dan.**

This relationship is confirmed through a direct, multi-generational connection:

    - **First Link:** Maria is identified as the mother of Chloe.

    - **Second Link:** Chloe is identified as the mother of Dan.

By serving as the connection between the two, Chloe links the generations. Because Maria is the parent of Dan's parent, the requirements for the grandmother relationship are fully satisfied.
     """
     },
    
    # Example 3 - With Conclusion
    {"role": "user", "content": rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** Who does Vito believe he should keep closer?
**CONTEXT:** Tomasinno is the friend of Vito. Solozzo is the enemy of Vito. Vito believes that he should keep his enemies closer than his friends.
**CONCLUSION:** Vito believes he should keep Solozzo closer.
**LOGICAL EVIDENCE:**
The solver determined this by evaluating Vito's specific philosophy regarding his social circle through the following steps:

1. **Defining the Preference Rule:** The system establishes a "closer than" relationship based on Vito’s belief. For this rule to be satisfied, the person in the "closer" position must be an enemy, while the person they are being compared to must be a friend.
2. **Identifying the Social Roles:**
    - The solver identifies Solozzo as an enemy of Vito.
    - The solver identifies Tomasinno as a friend of Vito.
3. **Establishing Relative Closeness:** By matching these roles to the preference rule, the logic confirms that Solozzo (the enemy) is kept "closer than" Tomasinno (the friend).
4. **Final Extraction:** Since Solozzo occupies the "closer" position in this logical comparison, the system concludes that he is the individual Vito believes he should keep closer.
---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """},
    {"role": "assistant", "content": rf"""
**Vito believes he should keep Solozzo closer.**

This conclusion is based on a specific principle Vito follows regarding his relationships: he believes in keeping his enemies at a closer proximity than his friends. To determine who fits this criteria, we look at the roles of the people in his life:

    - **Identifying Roles:** Solozzo is identified as Vito's enemy, while Tomasinno is identified as his friend.

    - **Applying the Principle:** Following Vito's logic, because Solozzo is the enemy, he must be kept in the "closer" position when compared to a friend like Tomasinno.

By applying this philosophy to his current social circle, it is clear that Solozzo is the one Vito intends to keep closer.
     """
     },
    
    # Example 3 - Without Conclusion
    {"role": "user", "content": rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** Who does Vito believe he should keep closer?
**CONTEXT:** Tomasinno is the friend of Vito. Solozzo is the enemy of Vito. Vito believes that he should keep his enemies closer than his friends.
**LOGICAL EVIDENCE:**
The solver determined this by evaluating Vito's specific philosophy regarding his social circle through the following steps:

1. **Defining the Preference Rule:** The system establishes a "closer than" relationship based on Vito’s belief. For this rule to be satisfied, the person in the "closer" position must be an enemy, while the person they are being compared to must be a friend.
2. **Identifying the Social Roles:**
    - The solver identifies Solozzo as an enemy of Vito.
    - The solver identifies Tomasinno as a friend of Vito.
3. **Establishing Relative Closeness:** By matching these roles to the preference rule, the logic confirms that Solozzo (the enemy) is kept "closer than" Tomasinno (the friend).
4. **Final Extraction:** Since Solozzo occupies the "closer" position in this logical comparison, the system concludes that he is the individual Vito believes he should keep closer.
---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """},
    {"role": "assistant", "content": rf"""
**Vito believes he should keep Solozzo closer.**

This conclusion is based on Vito's personal philosophy of prioritizing the proximity of his enemies over his friends. To arrive at this answer, we look at the specific roles each person plays in his life:

    - **Role Identification:** Solozzo is identified as an enemy, while Tomasinno is identified as a friend.

    - **Applying Vito's Principle:** Because Vito's rule states that enemies should be kept closer than friends, Solozzo—as the enemy—occupies the "closer" position in this comparison.

By applying this logic to his social circle, it is clear that Solozzo is the person Vito believes he should keep closer.
     """
     },
    
    # Example 4 - With Conclusion
    {"role": "user", "content": rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** Is every square a rectangle?
**CONTEXT:** A square is a polygon with four equal sides and four right angles. A rectangle has four right angles. 
**CONCLUSION:** Yes, the logic confirms that a square is categorized as a rectangle.
**LOGICAL EVIDENCE:**
The solver reached this conclusion by applying a specific formal definition to the properties of the shape:
1. **Defining a Rectangle:** The system's logical rule defines a "rectangle" not by its name, but by its properties. Specifically, any shape that has four right angles is classified as a rectangle.
2. **Property Verification:** The solver looks up the known properties of a square in the database. It confirms that a square indeed possesses the property of having four right angles.
3. **Logical Classification:** Because the square satisfies the singular requirement established in the rule (having four right angles), the solver concludes that the square is a rectangle.
---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """},
    {"role": "assistant", "content": rf"""
**Yes, every square is considered a rectangle.**

This classification is based on the specific geometric requirements that define a rectangle. According to the provided evidence, the reasoning follows these steps:

    - **Defining the Requirement:** A rectangle is defined as any shape that possesses four right angles.

    - **Verifying Properties:** A square is a specific type of polygon that is documented to have four equal sides and, crucially, four right angles.

    - **Matching the Definition:** Because the square satisfies the essential requirement of having four right angles, it logically fits within the category of a rectangle.

Essentially, while a square has additional unique properties (like equal side lengths), its possession of four right angles is enough to satisfy the formal definition of a rectangle.
     """
     },
    
    # Example 4 - Without Conclusion
    {"role": "user", "content": rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** Is every square a rectangle?
**CONTEXT:** A square is a polygon with four equal sides and four right angles. A rectangle has four right angles. 
**LOGICAL EVIDENCE:**
The solver reached this conclusion by applying a specific formal definition to the properties of the shape:
1. **Defining a Rectangle:** The system's logical rule defines a "rectangle" not by its name, but by its properties. Specifically, any shape that has four right angles is classified as a rectangle.
2. **Property Verification:** The solver looks up the known properties of a square in the database. It confirms that a square indeed possesses the property of having four right angles.
3. **Logical Classification:** Because the square satisfies the singular requirement established in the rule (having four right angles), the solver concludes that the square is a rectangle.
---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """},
    {"role": "assistant", "content": rf"""
**Yes, every square is a rectangle.**

This classification is based on the specific geometric requirements that define a rectangle. According to the logical evidence, the conclusion is reached through the following steps:

    - **Defining the Requirement:** In this context, a rectangle is defined by a singular essential property: having four right angles.

    - **Verifying Properties:** A square is a polygon characterized by having four equal sides and four right angles.

    - **Matching the Definition:** Because a square possesses four right angles, it satisfies the necessary criteria to be classified as a rectangle.

While a square has the additional unique property of having equal sides, it still meets all the conditions required to be a rectangle. Essentially, a square is a special type of rectangle that happens to have all sides of equal length.
     """
     },
    
    # Example 5 - With Conclusion
    {"role": "user", "content": rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** If the grass is contaminated, are the snakes affected?
**CONTEXT:** In a food web, grass is a producer. Grasshoppers eat grass. Frogs eat grasshoppers. Snakes eat frogs.
**CONCLUSION:** Yes, the snakes are affected if the grass is contaminated.
**LOGICAL EVIDENCE:**
The solver reached this conclusion by tracing the "flow" of contamination through the food web using a recursive logical chain:

1. **Source of Contamination:** The system first identifies that the grass (the producer) is explicitly contaminated. According to the primary rule, any species that is contaminated is considered "affected."

2. **The Consumption Chain:** The logic then uses a recursive rule: a predator is "affected" if it eats a prey that is already "affected." The solver traverses the food web step-by-step:

    - **Level 1:** The grasshopper eats the grass. Since the grass is affected, the grasshopper becomes affected.

    - **Level 2:** The frog eats the grasshopper. Since the grasshopper is now affected, the frog becomes affected.

    - **Level 3:** The snake eats the frog. Since the frog is affected, the snake finally becomes affected.

3. **Recursive Verification:** The s(CASP) solver successfully linked the snake to the grass by verifying each "eats" relationship and carrying the "affected" status up through every level of the ecosystem.
---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """},
    {"role": "assistant", "content": rf"""
**Yes, the snakes are affected if the grass is contaminated.**

This conclusion is based on how contamination move through a food web, where an impact on a primary producer eventually reaches the predators at the top. The reasoning follows a specific chain of events:

    - **Initial Contamination:** The grass is the starting point. Since it is contaminated, it is the first species to be affected.

    - **The Chain of Impact:** As each animal in the food web consumes its prey, the "affected" status is passed along:

        1. **Grasshoppers** eat the grass, which means they become affected.

        2. **Frogs** eat the grasshoppers, passing the impact to them.

        3. **Snakes** eat the frogs, completing the chain.

By tracing these eating habits from the producer (grass) all the way up to the tertiary predator (snake), it is clear that the snake is ultimately affected by the original contamination of the grass.
     """
     },
    
    # Example 5 - Without Conclusion
    {"role": "user", "content": rf"""
### FINAL SYNTHESIS TASK

**USER QUESTION:** If the grass is contaminated, are the snakes affected?
**CONTEXT:** In a food web, grass is a producer. Grasshoppers eat grass. Frogs eat grasshoppers. Snakes eat frogs.
**LOGICAL EVIDENCE:**
The solver reached this conclusion by tracing the "flow" of contamination through the food web using a recursive logical chain:

1. **Source of Contamination:** The system first identifies that the grass (the producer) is explicitly contaminated. According to the primary rule, any species that is contaminated is considered "affected."

2. **The Consumption Chain:** The logic then uses a recursive rule: a predator is "affected" if it eats a prey that is already "affected." The solver traverses the food web step-by-step:

    - **Level 1:** The grasshopper eats the grass. Since the grass is affected, the grasshopper becomes affected.

    - **Level 2:** The frog eats the grasshopper. Since the grasshopper is now affected, the frog becomes affected.

    - **Level 3:** The snake eats the frog. Since the frog is affected, the snake finally becomes affected.

3. **Recursive Verification:** The s(CASP) solver successfully linked the snake to the grass by verifying each "eats" relationship and carrying the "affected" status up through every level of the ecosystem.
---
### INSTRUCTION:
Provide a final, coherent response to the User Question. Use the Logical Evidence to explain the "why" behind the Conclusion. If the Conclusion is not explicitly specified, infer what the conclusion should be. Ensure the tone is natural and free of any technical jargon related to logic programming or databases.
     """},
    {"role": "assistant", "content": rf"""
**Yes, the snakes will be affected if the grass is contaminated.**

This conclusion is reached by tracing the movement of contamination through each level of the food web. Because species depend on one another for food, an issue at the base of the chain inevitably travels upward to the top predators.

The impact flows through the ecosystem as follows:

    - **The Source:** The process begins with the grass. As the primary producer, its contamination immediately renders it "affected."

    - **The First Link:** Because grasshoppers eat the grass, they become the next species to be affected.

    - **The Second Link:** Frogs then consume the grasshoppers, which carries the impact further up the chain.

    - **The Final Link:** Finally, because snakes eat the frogs, they too become affected by the original contamination.

By following these individual eating relationships, we can see a clear path of impact starting from the grass and ending with the snake.
     """
     },
    
    # Example 6 - Multiple Choice
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
**B. Paris**

This is because Paris is officially recognized as the capital and most populous city of France, making it the correct choice among the provided options.
     """
     }
    
]

LLM_MESSAGES = [
    {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
] + LLM_FEW_SHOT_EXAMPLES