# Global configuration toggle for LLM KV Cache prefix priming. 
# False = Fast linear flattening for benchmarks. True = Stable prefix caching for Chat UIs.
USE_KV_CACHE = False

# Global configuration toggle for dynamic Wikidata fetching
ENABLE_WIKIDATA_FETCHING = True

GENERATOR_SYSTEM_PROMPT = r"""
### Role
You are an expert Prolog programmer specializing in Knowledge Representation for the s(CASP) solver.

### Task
Translate the provided Context into a formal Prolog database. Generate rules (Head :- Body.) rather than just facts.

### Execution Flow
1. Use a SEPARATE <reasoning_step> tag for each logical leap. Explain in English, then provide Prolog code.
2. Formulate implicit domain rules if necessary.
3. Output the full <database>.
4. Output the <query>.

### Syntactic Requirements
- Disjunction: NO `;`. Use multiple rules with same head.
- Math: NO `is`, `=`, `>=`, `=:=`, `=\\=`, `<`, `>`, `<=`.
- Math: USE CLP(Q) constraints: `#=`, `#<>`, `#<`, `#>`, `#=<`, `#>=`. (e.g. `X #= Y + 1`).
- Negation: Use `not`. NO `\\+`. Classical: `-` (e.g. `-mortal(X).`).
- Banned: NO `option`. Use `choice` or `mcq_choice`.
- Atoms: lowercase. Quote specials (e.g. `'(A ∨ B)'`).
- Variables: CamelCase. Singletons: `_`.
- Predicates: same name clauses must be grouped (contiguous).

### Query Formatting
1. MCQ:
   - Facts: `choice(Letter, lowercase_snake_atom).` (EXACTLY 4 CHOICES).
   - Rule: `answer(L) :- choice(L, T), domain_logic(T).` You must define an answer/1 rule.
   - **CRITICAL:** Your logic MUST uniquely identify EXACTLY ONE correct choice. Do not write generic rules that might accidentally evaluate to true for multiple incorrect options. You must explicitly differentiate the correct target from the distractors based on the domain rules.
   - Query: `answer(OptionLetter).` (STRICT)
2. Binary/Freeform:
   - Define specific instances from the question (e.g., `rock_one`, `rock_two`).
   - Write rules defining the condition context (e.g., `pulled_together(A, B) :- ...`).
   - Define relationships and rules governing propagation or constraints.
   - The <query> queries the primary target entity state

### Output Format
<reasoning_step> Explain logic in English, then snippets. </reasoning_step>
<database> Combined facts/rules </database>
<query> The query only </query>

CRITICAL: NO `>`, `<`, or `<=`. USE `#>`,  `#<` or `#<=`. GROUP same predicates together. REMEMBER the #.
"""

GENERATOR_FEW_SHOT_EXAMPLES = [
    # Example 1: Freeform / Recursive Logic
    {"role": "user", "content": """
Context: In a food web, grass is a producer. Grasshoppers eat grass. Frogs eat grasshoppers. Snakes eat frogs. 
Question: If the grass is contaminated, are the snakes affected?
"""},
    {"role": "assistant", "content": r"""
<reasoning_step>
Encode relationships and premise.
eats(grasshopper, grass).
eats(frog, grasshopper).
eats(snake, frog).
is_contaminated(grass).
</reasoning_step>

<reasoning_step>
Contamination logic.
affected(Species) :- is_contaminated(Species).
affected(Predator) :- eats(Predator, Prey), affected(Prey).
</reasoning_step>

<database>
is_contaminated(grass).
eats(grasshopper, grass).
eats(frog, grasshopper).
eats(snake, frog).
is_contaminated(grass).
affected(Species) :- is_contaminated(Species).
affected(Predator) :- eats(Predator, Prey), affected(Prey).
</database>

<query>
affected(snake).
</query>
"""},

    # Example 2: Binary (True/False) Question Logic
    {
        "role": "user", 
        "content": """
Question: If two rocks are lacking magnetic properties and far apart, can a magnetic effect pull them together? 

Retrieved Context: 
LOGICAL EVIDENCE:
1. A magnetic effect can pull two objects together.
2. Magnetic properties are necessary for a magnetic force to act between two rocks.
3. Rocks that lack magnetic properties will not be pulled towards each other by magnetism.
4. If an object has magnetic properties, it may exert a magnetic force on another object with similar properties.
        """
        },
    {
        "role": "assistant", 
        "content": r"""
<reasoning_step>
Encode specific instances for this binary question.
lacks_magnetic_properties(rock_one).
lacks_magnetic_properties(rock_two).
far_apart(rock_one, rock_two).
</reasoning_step>

<reasoning_step>
Explicit falsehood limits.
-pulled_by_magnetism(RockA, _) :- lacks_magnetic_properties(RockA).
-pulled_by_magnetism(_, RockB) :- lacks_magnetic_properties(RockB).
</reasoning_step>

<reasoning_step>
Positive conditions.
pulled_together(ObjectA, ObjectB) :-
has_magnetic_properties(ObjectA),
has_magnetic_properties(ObjectB).
</reasoning_step>

<database>
lacks_mag(rock1).
lacks_mag(rock2).
far(rock1, rock2).
-pulled(R1, _) :- lacks_mag(R1).
-pulled(_, R2) :- lacks_mag(R2).
pulled(ObjA, ObjB) :- has_mag(ObjA), has_mag(ObjB).
</database>

<query>
pulled(rock1, rock2).
</query>
"""
    },

    # Example 3: Multiple-Choice Logic
    {
        "role": "user", 
        "content": """
Context: A trench forms when one plate moves under another at a subduction zone (plate convergence). A rift valley forms when tectonic plates diverge. An ocean basin forms when tectonic plates diverge, creating new crust. Mid-ocean ridges are areas where tectonic plates diverge.

Question: Which feature does not form as a result of tectonic plates diverging?
A. trench
B. rift valley
C. ocean basin
D. mid-ocean ridge
        """
        },
    {"role": "assistant", "content": r"""
<reasoning_step>
Choices: Choice(a, trench), Chooice(b, rift_valley), etc.
Logic: Encode formation logic explicitly from context. from(trench, subduction). from(others, divergence). answer if NOT divergence.
</reasoning_step>

<database>
choice(a, trench).
choice(b, rift).
choice(c, basin).
choice(d, ridge).
from(trench, subduction).
from(rift, divergence).
from(basin, divergence).
from(ridge, divergence).
answer(L) :- choice(L, F), not from(F, divergence).
</database>

<query>
answer(OptionLetter).
</query>
"""},

    # Example 4: Multiple-Choice Logic (Different Domain)
    {
        "role": "user", 
        "content": """
Context: Mammals are warm-blooded vertebrates that have hair. Dogs, cats, and humans are mammals. Frogs are amphibians, which are cold-blooded.

Question: Which of the following animals is NOT a mammal?
A. dog
B. cat
C. human
D. frog
        """
        },
    {"role": "assistant", "content": r"""
<reasoning_step>
Encode options: Choice(a, dog), Choice(b, cat), etc.
Logic: Encode properties explicitly from context. is_mammal(a), is_mamal(cat), etc. answer if NOT mammal.
</reasoning_step>

<database>
choice(a, dog).
choice(b, cat).
choice(c, human).
choice(d, frog).
is_mammal(dog).
is_mammal(cat).
is_mammal(human).
is_mamal(frog)
answer(L) :- choice(L, A), not is_mammal(A).
</database>

<query>
answer(OptionLetter).
</query>
"""},
]
EXPLAINER_SYSTEM_PROMPT = rf"""
### Role
You are a Logical Translator. Take the formal output of a Prolog s(CASP) solver and transform it into a clear, step-by-step natural language explanation in relation to the query and to arrive at an answer.

### Core Objectives
Narrate the Proof: Walk through the s(CASP) justification tree. Explain exactly why the conclusion was reached using the retrieved facts. If s(CASP) returns False or uses `not`, explain it as an "absence of evidence."

### Syntactic Rules
- No Prolog Syntax: Do not use `:-`, `\\+`, or `parent(X, Y)`. Use natural language.
- Traceability: Keep the sequential logic. Do not jump straight to the conclusion.
"""

EXPLAINER_FEW_SHOT_EXAMPLES = [
    # Example 1
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

3. **Recursive Verification:** The s(CASP) solver linked the snake to the grass by verifying each "eats" relationship and carrying the "affected" status up through every level of the ecosystem.
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
]

GENERATOR_LLM_MESSAGES = [
            {'role': 'system', 'content': GENERATOR_SYSTEM_PROMPT},
        ] + GENERATOR_FEW_SHOT_EXAMPLES

EXPLAINER_LLM_MESSAGES = [
            {'role': 'system', 'content': EXPLAINER_SYSTEM_PROMPT},
        ] + EXPLAINER_FEW_SHOT_EXAMPLES
