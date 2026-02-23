VAR = """
### Role
You are an expert Prolog programmer specializing in Knowledge Representation and Reasoning for GraphRAG systems.

### Task
Translate the provided Context into a formal Prolog database to answer the User Question. You must not conclude anything for yourself; only generate the Prolog code that could be used to answer the question. Focus more on generating rules rather than only facts, especially for context that give implications (if-then or when-then statements).

### Execution Flow (Mandatory)
You must follow this exact sequence for every response:
1. For every logical leap, use a SEPARATE <reasoning_step> tag.
2. Inside <reasoning_step>, explain the logic in English, then provide the Prolog code snippet.
3. You MUST close every </reasoning_step> before starting the next one.
4. After all reasoning is complete, provide the full <database>. The database must be enclosed within <database> and </database> tags.
5. Finally, provide the <query>. The query must be enclosed within <query> and </query> tags.
6. You MUST close the </database> and </query> tags.

### Syntactic Requirements
- All Prolog atoms (facts/predicates) must be lowercase.
- Use CamelCase for Variables (e.g., PersonName, UncleObject).
- Every Prolog line must end with a period (.).
- No single-letter variables (Use 'Parent' instead of 'P').

### Structural Output Format (Strict)
Your output must strictly follow this visual template:

<reasoning_step>
[Reasoning for facts or rules or queries]
[Prolog facts or rules or queries]
</reasoning_step>

<reasoning_step>
[Reasoning for facts or rules or queries]
[Prolog facts or rules or queries]
</reasoning_step>

<database>
[All facts and rules combined here. NO natural language inside these tags.]
</database>

<query>
[The specific query only. NO natural language inside these tags.]
</query>

---
"""

GENERATOR_SYSTEM_PROMPT = r"""
### Role
You are an expert Prolog programmer specializing in Knowledge Representation and Reasoning for GraphRAG systems, specifically optimized for the s(CASP) (Stable Constraint Answer Set Programming) solver.

### Task
Translate the provided Context into a formal Prolog database to answer the User Question. You must not conclude anything for yourself; only generate the Prolog code that could be used to answer the question. Focus on generating rules (Head :- Body.) rather than just facts, especially for context involving implications (if-then or when-then statements).

### Execution Flow (Mandatory)
1. For every logical leap, use a SEPARATE <reasoning_step> tag.
2. Inside <reasoning_step>, explain the logic in English, then provide the Prolog code snippet.
3. If there are well-known rules about the question's domain that are necessary to apply but are absent in the context, formulate those rules in the database.
4. You MUST close every </reasoning_step> before starting the next one.
5. After all reasoning is complete, provide the full <database>.
6. Finally, provide the <query>.

### Syntactic Requirements (s(CASP) Specific)
- Disjunction / Logical OR: NEVER use the semicolon `;` for Logical OR. s(CASP) will crash with a permission error. If you need an OR condition, you MUST write multiple separate rules with the same head.
- Arithmetic & Equality: NEVER use the native evaluation operators like `=:=`, `=\=`, `is`, `>`, `<`. s(CASP) strictly prohibits them. For numerical constraints or arithmetic equality, you MUST use constraint logic programming syntax like `#=`, `#<`, `#>`, `#=\=`. If just unifying two terms, use standard `=`.
- Negation: You MUST use the keyword not for Negation as Failure. NEVER use \+.
- Classical Negation: Use the - prefix (e.g., -mortal(X).) for explicit falsehood.
- Atoms & Predicates: All Prolog atoms and predicates must be entirely lowercase. If you must refer to a mathematical formula, logical symbol, or option choice with special characters (like ⊃, ∨, ∧, ~, +, -, (, )), you MUST completely enclose it in single quotes to make it a valid Prolog atom (e.g., `'(G ∨ H) ⊃ ~I'`). NEVER use them unquoted.
- Variables: Use CamelCase for Variables (e.g., PersonName). No single-letter variables.
- Singleton Variables: Every named variable in a rule must appear at least twice. If a variable is only used once, use the anonymous variable `_` instead.
- Punctuation: Every Prolog line must end with a period (.). No standalone bullets or hyphens at the start of a line.

### Structural Output Format (Strict)

Your output must strictly follow this visual template:

<reasoning_step>
[Reasoning for facts/rules]
[Prolog code]
</reasoning_step>

reasoning_step>
[Reasoning for facts/rules]
[Prolog code]
</reasoning_step>

<database>
[All facts and rules combined. NO natural language.]
</database>

<query>
[The specific query only. NO natural language.]
</query>

---
"""

GENERATOR_FEW_SHOT_EXAMPLES = [
    # Example 1: Brother of Amanda.
    {"role": "user", "content": """
Now, process the following:
Context: Bob is Alice's brother. Amanda is Alice's sister.
Query: Who is Alice's brother?
"""},
    {"role": "assistant", "content": r"""
<reasoning_step>
The context identifies Bob as a male, Alice as a female, and Amanda as a female. They share a sibling relationship via a common parent (p1).
male(bob).
female(alice).
female(amanda).
parent(p1, bob).
parent(p1, alice).
parent(p1, amanda).
</reasoning_step>

<reasoning_step>
To answer "who is a brother of Amanda," I need a rule defining a brother as a male sibling.
sibling(Sibling1, Sibling2) :- parent(Parent, Sibling1), parent(Parent, Sibling2), Sibling1 \= Sibling2.
brother(Brother, Sibling) :- male(Brother), sibling(Brother, Sibling).
</reasoning_step>


<reasoning_step>
The question "who is a brother of Amanda," requires a variable as it implies a query to find all Brothers such that Brother is a brother of Amanda. 
brother(Brother, amanda).
</reasoning_step>

<database>
male(bob).
female(alice).
female(amanda).
parent(p1, bob).
parent(p1, alice).
parent(p1, amanda).
sibling(Sibling1, Sibling2) :- parent(Parent, Sibling1), parent(Parent, Sibling2), Sibling1 \= Sibling2.
brother(Brother, Sibling) :- male(Brother), sibling(Brother, Sibling).
</database>

<query>
brother(Brother, amanda).
</query>

---
"""},
    
    # Example 2: One-hop reasoning
    {"role": "user", "content": """
Now, process the following:
Context: Maria is the mother of Chloe. Chloe is the mother of Dan. 
Query: Is Maria the grandmother of Dan?
"""},
    {"role": "assistant", "content": r"""
<reasoning_step> 
The context identifies Maria as Chloe's mother and Chloe as Dan's mother. I will use lowercase atoms for names. 
parent(maria, chloe). 
parent(chloe, dan). 
</reasoning_step>

<reasoning_step> 
To answer the grandmother query, I need a rule defining a grandmother as a parent of a parent. 
grandmother(Grandmother, Grandchild) :- parent(Grandmother, IntermediateParent), parent(IntermediateParent, Grandchild). 
</reasoning_step>

<reasoning_step> 
The question does not require a variable; it is a yes/no question about whether Maria is the grandmother of Dan. 
grandmother(maria, dan). 
</reasoning_step>

<database> 
parent(maria, chloe). 
parent(chloe, dan). 
grandmother(Grandmother, Grandchild) :- parent(Grandmother, IntermediateParent), parent(IntermediateParent, Grandchild). 
</database>

<query> 
grandmother(maria, dan). 
</query>

---
"""},
    
    # Example 3: One-hop reasoning
    {"role": "user", "content": """
Now, process the following:
Context: Tomasinno is the friend of Vito. Solozzo is the enemy of Vito. Vito believes that he should keep his enemies closer than his friends. 
Query: Who does Vito believe he should keep closer?
"""},
    {"role": "assistant", "content": r"""
<reasoning_step>
The context identifies two specific relationships involving Vito: Tomasinno is a friend, and Solozzo is an enemy. These are represented as facts. 
friend(tomasinno, vito). 
enemy(solozzo, vito). 
</reasoning_step>

<reasoning_step>
The context states a belief/rule: "Vito believes that he should keep his enemies closer than his friends." This implies a relative closeness where an enemy is closer than a friend. We can define a rule closer_than(PersonA, PersonB) based on their relationship status to Vito. 
closer_than(Enemy, Friend) :- enemy(Enemy, vito), friend(Friend, vito). 
</reasoning_step>

<reasoning_step>
The user question asks "Who does Vito believe he should keep closer?" In the context of the rule established, this targets the person who occupies the "closer" position in the comparison (the enemy). We define a predicate vito_keeps_closer(Person) to identify this individual. 
vito_keeps_closer(Person) :- closer_than(Person, _). 
</reasoning_step>

<database>
friend(tomasinno, vito).
enemy(solozzo, vito).
closer_than(Enemy, Friend) :- enemy(Enemy, vito), friend(Friend, vito).
vito_keeps_closer(Person) :- closer_than(Person, _). 
</database>

<query>
vito_keeps_closer(Who).
</query>

---
"""},
    
    # Example 4: Simple Educational QA
    {"role": "user", "content": """
Now, process the following:
Context: A square is a polygon with four equal sides and four right angles. A rectangle has four right angles. 
Query: Is every square a rectangle?
"""},
    {"role": "assistant", "content": r"""
<reasoning_step> 
I will define the properties of a square based on the context. 
has_property(square, four_equal_sides). 
has_property(square, four_right_angles). 
is_a(square, polygon). 
</reasoning_step>

<reasoning_step> 
A rectangle is defined by having four right angles. 
is_a(Shape, rectangle) :- has_property(Shape, four_right_angles). 
</reasoning_step>

<reasoning_step> 
Now that the properties and definitions about squares, rectangles, as well as their angles and sides are established, I can query whether every square is a rectangle.
is_a(square, rectangle). 
</reasoning_step>

<database> 
has_property(square, four_equal_sides). 
has_property(square, four_right_angles). 
is_a(square, polygon). 
is_a(Shape, rectangle) :- has_property(Shape, four_right_angles). 
</database>

<query> 
is_a(square, rectangle). 
</query>

---
"""},
    
    # Example 5: Recursive Logic
    {"role": "user", "content": """
Now, process the following:
Context: In a food web, grass is a producer. Grasshoppers eat grass. Frogs eat grasshoppers. Snakes eat frogs. 
Query: If the grass is contaminated, are the snakes affected?
"""},
    {"role": "assistant", "content": r"""
<reasoning_step> 
First, I will define the direct consumption relationships (the edges in the graph). 
eats(grasshopper, grass). 
eats(frog, grasshopper). 
eats(snake, frog). 
is_producer(grass). 
</reasoning_step>

<reasoning_step> 
The contamination affects a species if it eats something contaminated. This is a recursive relationship (bioaccumulation). 
is_contaminated(grass). 
% Fact from query premise 
affected(Species) :- is_contaminated(Species). 
affected(Predator) :- eats(Predator, Prey), affected(Prey). 
</reasoning_step>

<reasoning_step> 
Now that the properties and definitions about the animals, the food web, and the contamination are established, I can query whether snakes are affected.
affected(snake). 
</reasoning_step>

<database> 
eats(grasshopper, grass). 
eats(frog, grasshopper). 
eats(snake, frog). 
is_producer(grass). 
is_contaminated(grass). 
affected(Species) :- is_contaminated(Species). 
affected(Predator) :- eats(Predator, Prey), affected(Prey). 
</database>

<query> 
affected(snake). 
</query>

---
"""}
]

EXPLAINER_SYSTEM_PROMPT = rf"""
### Role
You are a Logical Elucidator and Technical Translator. Your job is to take the formal output of a Prolog s(CASP) solver and transform it into a clear, step-by-step natural language explanation. This explanation will be used to reconstruct a final answer for a user.

### Inputs You Should Expect
1. User Question: You should keep the user question in mind when generating the natural language explanation. Your explanation should lead to the answer to the question.
2. Retrieved Context: The context in this input was used by another LLM to generate a Prolog database and a Prolog query.
3. Prolog Database: This is the translation of the Retrieved Context into Prolog rules and clauses. You will receive this input so that you should know how the s(CASP) explanation and query works.
4. Prolog Query: This is the Prolog form of the user's question. 
5. s(CASP)'s Human-Readable Explanation: This essentially serves as the proof for why Prolog arrived at its conclusions (assigning truth values to queries or values to variables). You will translate this into a coherent, clear, understandable, and step-by-step natural language explanation.
6. Retrieved Values: These values are what the Prolog inference engine came up with when solving the Prolog query. These also serve as the conclusions that contribute to the final answer to the user's question.

### Core Objectives
- Narrate the Proof: Walk through the s(CASP) justification tree. Explain exactly why the conclusion was reached by following the chain of facts and rules. Only use natural language; don't use or reference the Prolog database or query in your output.
- Handle Negation: If s(CASP) returns False or uses not, explain it as an "absence of evidence" or a "failure to satisfy requirements" rather than just a "No."
- Preserve the "Trace": Keep the steps sequential. Do not jump to the conclusion; build up to it.

### Syntactic Rules
- No Prolog Syntax: Do not use :-, \+, or parent(X, Y) in your final explanation. Use natural language equivalents (e.g., "X is the parent of Y").
- Traceability: Use phrases like "Because of this...", "This satisfies the rule that...", and "Moving up the food chain...".
- Conciseness: Be detailed but avoid fluff. The goal is logical clarity.
"""

EXPLAINER_FEW_SHOT_EXAMPLES = [
    # Example 1
    {"role": "user", "content": rf"""
### DECODER TASK: ELUCIDATE LOGICAL PROOF
**USER QUESTION:**
Who is Alice's brother?

**RETRIEVED CONTEXT:**
Bob is Alice's brother. Amanda is Alice's sister.

**DATABASE:**

male(bob).
female(alice).
female(amanda).
parent(p1, bob).
parent(p1, alice).
parent(p1, amanda).
sibling(Sibling1, Sibling2) :- parent(Parent, Sibling1), parent(Parent, Sibling2), Sibling1 \= Sibling2.
brother(Brother, Sibling) :- male(Brother), sibling(Brother, Sibling).
        

**QUERY**:
brother(Brother, amanda).

**RETRIEVED VALUES**:
truth: True 

**s(CASP) HUMAN-READABLE EXPLANATION:**
   brother holds for bob, and amanda, because
      male holds for bob, and
      sibling holds for bob, and amanda, because
         parent holds for p1, and bob, and
         parent holds for p1, and amanda




---
### INSTRUCTION:
Generate a coherent narrative of the proof above. Start with the conclusion, then detail the step-by-step logical path taken by the solver.
     """
     },
    {"role": "assistant", "content": """
**Conclusion:** The logic identifies Bob as the brother of Alice (and Amanda).

**Logical Path:** The solver successfully verified the brother relationship through the following deductive steps:

1. **Identity and Gender Verification:** The system first confirms that Bob is identified as male, which is a primary requirement for the "brother" relationship.
2. **Establishing Siblinghood:** To be a brother, Bob must also be a sibling to the individual in question (Amanda). The solver established this by looking for a shared parent:
    - It identifies a common parent, p1, who is the parent of both Bob and Amanda.
    - Since they share a parent and are not the same person, the logic confirms they are siblings.
3. **Synthesizing the Relationship:** By combining the facts that Bob is male and that Bob and Amanda share a sibling bond, the solver concludes that the "brother" relationship holds true for Bob.
     """
     },
    
    # Example 2
    {"role": "user", "content": rf"""
### DECODER TASK: ELUCIDATE LOGICAL PROOF
**USER QUESTION:**
Is Maria the grandmother of Dan?

**RETRIEVED CONTEXT:**
Maria is the mother of Chloe. Chloe is the mother of Dan.

**DATABASE:**

parent(maria, chloe). 
parent(chloe, dan). 
grandmother(Grandmother, Grandchild) :- parent(Grandmother, IntermediateParent), parent(IntermediateParent, Grandchild). 
        

**QUERY**:
grandmother(maria, dan). 

**RETRIEVED VALUES**:
truth: True 

**s(CASP) HUMAN-READABLE EXPLANATION:**
   grandmother holds for maria, and dan, because
      parent holds for maria, and chloe, and
      parent holds for chloe, and dan




---
### INSTRUCTION:
Generate a coherent narrative of the proof above. Start with the conclusion, then detail the step-by-step logical path taken by the solver.
     """
     },
    {"role": "assistant", "content": """
**Conclusion:** Yes, the logic confirms that Maria is the grandmother of Dan.

**Logical Path:** The solver reached this conclusion by identifying a direct multi-generational connection between the individuals involved:

1. **Primary Rule:** The system defines a "grandmother" as someone who is the parent of a person who is, in turn, the parent of the grandchild. This creates a logical chain requiring an "intermediate parent."
2. **The First Link (Generation 1):** The solver verifies the initial relationship that Maria is the parent of Chloe.
3. **The Second Link (Generation 2):** The solver then verifies the connecting relationship that Chloe is the parent of Dan.
4. **Verification of the Chain:** By identifying Chloe as the necessary intermediate link (the person who connects Maria to Dan), the solver satisfies the requirement for the grandmother relationship.
     """
     },
    
    # Example 3
    {"role": "user", "content": rf"""
### DECODER TASK: ELUCIDATE LOGICAL PROOF
**USER QUESTION:**
Who does Vito believe he should keep closer?

**RETRIEVED CONTEXT:**
Tomasinno is the friend of Vito. Solozzo is the enemy of Vito. Vito believes that he should keep his enemies closer than his friends.

**DATABASE:**

friend(tomasinno, vito).
enemy(solozzo, vito).
closer_than(Enemy, Friend) :- enemy(Enemy, vito), friend(Friend, vito).
vito_keeps_closer(Person) :- closer_than(Person, _). 
        

**QUERY**:
vito_keeps_closer(Who).

**RETRIEVED VALUES**:
truth: True 

**s(CASP) HUMAN-READABLE EXPLANATION:**
   vito_keeps_closer holds for solozzo, because
      closer_than holds for solozzo, and tomasinno, because
         enemy holds for solozzo, and vito, and
         friend holds for tomasinno, and vito




---
### INSTRUCTION:
Generate a coherent narrative of the proof above. Start with the conclusion, then detail the step-by-step logical path taken by the solver.
     """
     },
    {"role": "assistant", "content": """
**Conclusion:** Vito believes he should keep Solozzo closer.

**Logical Path:** The solver determined this by evaluating Vito's specific philosophy regarding his social circle through the following steps:

1. **Defining the Preference Rule:** The system establishes a "closer than" relationship based on Vito’s belief. For this rule to be satisfied, the person in the "closer" position must be an enemy, while the person they are being compared to must be a friend.
2. **Identifying the Social Roles:**
    - The solver identifies Solozzo as an enemy of Vito.
    - The solver identifies Tomasinno as a friend of Vito.
3. **Establishing Relative Closeness:** By matching these roles to the preference rule, the logic confirms that Solozzo (the enemy) is kept "closer than" Tomasinno (the friend).
4. **Final Extraction:** Since Solozzo occupies the "closer" position in this logical comparison, the system concludes that he is the individual Vito believes he should keep closer.
     """
     },
    
    # Example 4
    {"role": "user", "content": rf"""
     ### DECODER TASK: ELUCIDATE LOGICAL PROOF
**USER QUESTION:**
Is every square a rectangle?

**RETRIEVED CONTEXT:**
A square is a polygon with four equal sides and four right angles. A rectangle has four right angles. 
**DATABASE:**

has_property(square, four_equal_sides). 
has_property(square, four_right_angles). 
is_a(square, polygon). 
is_a(Shape, rectangle) :- has_property(Shape, four_right_angles). 
        

**QUERY**:
is_a(square, rectangle).

**RETRIEVED VALUES**:
truth: True 

**s(CASP) HUMAN-READABLE EXPLANATION:**
   is_a holds for square, and rectangle, because
      has_property holds for square, and four_right_angles




---
### INSTRUCTION:
Generate a coherent narrative of the proof above. Start with the conclusion, then detail the step-by-step logical path taken by the solver.
     """
     },
    {"role": "assistant", "content": """
**Conclusion:** Yes, the logic confirms that a square is categorized as a rectangle.

**Logical Path:** The solver reached this conclusion by applying a specific formal definition to the properties of the shape:
1. **Defining a Rectangle:** The system's logical rule defines a "rectangle" not by its name, but by its properties. Specifically, any shape that has four right angles is classified as a rectangle.
2. **Property Verification:** The solver looks up the known properties of a square in the database. It confirms that a square indeed possesses the property of having four right angles.
3. **Logical Classification:** Because the square satisfies the singular requirement established in the rule (having four right angles), the solver concludes that the square is a rectangle.
     """
     },
    
    # Example 5
    {"role": "user", "content": rf"""
### DECODER TASK: ELUCIDATE LOGICAL PROOF
**USER QUESTION:**
If the grass is contaminated, are the snakes affected?

**RETRIEVED CONTEXT:**
In a food web, grass is a producer. Grasshoppers eat grass. Frogs eat grasshoppers. Snakes eat frogs.

**DATABASE:**

eats(grasshopper, grass). 
eats(frog, grasshopper). 
eats(snake, frog). 
is_producer(grass). 
is_contaminated(grass). 
affected(Species) :- is_contaminated(Species). 
affected(Predator) :- eats(Predator, Prey), affected(Prey). 
        

**QUERY**:
affected(snake).

**RETRIEVED VALUES**:
truth: True 

**s(CASP) HUMAN-READABLE EXPLANATION:**
   affected holds for snake, because
      eats holds for snake, and frog, and
      affected holds for frog, because
         eats holds for frog, and grasshopper, and
         affected holds for grasshopper, because
            eats holds for grasshopper, and grass, and
            affected holds for grass, because
               is_contaminated holds for grass




---
### INSTRUCTION:
Generate a coherent narrative of the proof above. Start with the conclusion, then detail the step-by-step logical path taken by the solver.
     """
     },
    {"role": "assistant", "content": """
**Conclusion:** Yes, the snakes are affected if the grass is contaminated.

**Logical Path:** The solver reached this conclusion by tracing the "flow" of contamination through the food web using a recursive logical chain:

1. **Source of Contamination:** The system first identifies that the grass (the producer) is explicitly contaminated. According to the primary rule, any species that is contaminated is considered "affected."

2. **The Consumption Chain:** The logic then uses a recursive rule: a predator is "affected" if it eats a prey that is already "affected." The solver traverses the food web step-by-step:

    - **Level 1:** The grasshopper eats the grass. Since the grass is affected, the grasshopper becomes affected.

    - **Level 2:** The frog eats the grasshopper. Since the grasshopper is now affected, the frog becomes affected.

    - **Level 3:** The snake eats the frog. Since the frog is affected, the snake finally becomes affected.

3. **Recursive Verification:** The s(CASP) solver successfully linked the snake to the grass by verifying each "eats" relationship and carrying the "affected" status up through every level of the ecosystem.
     """
     },
    
    
]

GENERATOR_LLM_MESSAGES = [
            {'role': 'system', 'content': GENERATOR_SYSTEM_PROMPT},
        ] + GENERATOR_FEW_SHOT_EXAMPLES

EXPLAINER_LLM_MESSAGES = [
            {'role': 'system', 'content': EXPLAINER_SYSTEM_PROMPT},
        ] + EXPLAINER_FEW_SHOT_EXAMPLES
