# Global configuration toggle for LLM KV Cache prefix priming. 
# False = Fast linear flattening for benchmarks. True = Stable prefix caching for Chat UIs.
USE_KV_CACHE = True

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

### Syntactic Requirements (s(CASP) Specific)
- Disjunction: NEVER use `;`. Use multiple separate rules with the same head instead.
- Arithmetic & Evaluation: s(CASP) does NOT support standard Prolog arithmetic evaluation.
   - NEVER use `is`, , `>=`, `=:=`, `=\\=`, `<`, `>`, `<=`.
   - USE CLP(Q) constraints instead: `#=`, `#<>`, `#<`, `#>`, `#=<`, `#>=`. Example: `X #= Y + 1`.
- Negation: Use `not`. NEVER use `\\+`.
- Classical Negation: Use `-` prefix (e.g., `-mortal(X).`).
- Banned Predicates: NEVER use `option`. It conflicts with the SWI-Prolog standard library. Use `choice` or `mcq_choice` instead.
- Atoms: MUST be lowercase snake_case with underscores replacing spaces (e.g., `cell_wall`, `cell_membrane`). NEVER use spaces inside atoms — `choice(a, cell membrane)` is a SYNTAX ERROR. Write `choice(a, cell_membrane)` instead.
- Grounding: EVERY fact argument must be a lowercase atom. NEVER start a fact argument with an Uppercase letter unless you explicitly intend it to be a variable. CamelCase names (e.g., `PlantCell`) are treated as Variables and will trigger "Singleton variable" warnings.
- Variables: MUST start with an Uppercase letter (e.g., `X`, `Animal`). Use `_` for singletons.
- Meaningful Atoms: Use snake_case atoms that accurately reflect the phrasing in the text. Condense long phrases into 3-5 critical words (e.g., `eliminating_a_source_from_consideration` -> `eliminate_source`). The meaning must remain clear.
- Closed-World: EVERY predicate in a body MUST be defined as a fact or rule head somewhere in the database.
- Contiguous: Group ALL clauses for the same predicate together. Do NOT alternate between predicates (e.g., `p(a). q(b). p(c).` is forbidden; write `p(a). p(c). q(b).` instead).
- Predicate Naming: ALWAYS invent meaningful predicate names that accurately describe the relationships in the specific question. NEVER blindly copy predicate names (like `forms_from`) or structures from the few-shot examples unless they are directly relevant.

### Query Formatting by Question Type
1. **Multiple-Choice Questions (MCQ):**
   - Encode EXACTLY FOUR choices as facts. The description MUST be a grounded lowercase snake_case atom derived from the ACTUAL option text, NEVER an uppercase variable!
   - Example CORRECT: `choice(a, eliminating_a_source).`
   - Example WRONG: `choice(a, A).` (This will trivially unify with anything and fail the proof!)
   - Encode retrieved domain knowledge as facts and rules.
   - You MUST define an `answer/1` rule that unifies with the correct option letter.
   - **CRITICAL:** Your logic MUST uniquely identify EXACTLY ONE correct choice. Do not write generic rules that might accidentally evaluate to true for multiple incorrect options. You must explicitly differentiate the correct target from the distractors based on the domain rules.
   - Example DB Rule: `answer(OptionLetter) :- choice(OptionLetter, Target), matches_criteria(Target).`
   - The <query> MUST ALWAYS BE EXACTLY: `answer(OptionLetter).`
2. **Binary (True/False) Questions:**
   - Define specific instances from the question (e.g., `rock_one`, `rock_two`).
   - Write rules defining the condition context (e.g., `pulled_together(A, B) :- ...`).
   - The <query> MUST test the fully grounded relationship directly (e.g., `pulled_together(rock_one, rock_two).`).
3. **Freeform / Open Logic:**
   - Define relationships and rules governing propagation or constraints.
   - The <query> queries the primary target entity state.

### Warning Avoidance (CRITICAL)
- **NO SINGLETONS**: If you start a concept name with an Uppercase letter in a fact (e.g., `can_be_controlled(RecordingObservations).`), Prolog treats it as a variable. This triggers "Singleton variable" warnings and usually makes the logic fail. **ALWAYS use lowercase snake_case for facts.**
- **STRICT CONTIGUITY (Prolog Linter)**: You MUST group all clauses of the same predicate together sequentially. Do NOT alternate between different predicates. 
  - WRONG: `p(a). q(b). p(c).`
  - CORRECT: `p(a). p(c). q(b).`
- **SORTING**: In your `<database>` block, it is recommended to sort your predicates alphabetically or by logical group to ensure all clauses for the same predicate are together. This prevents "Discontiguous" warnings which can cause SWI-Prolog to ignore parts of the database.

### Output Format (Strict)
<reasoning_step>
[Reasoning in English]
[Prolog code snippets]
</reasoning_step>

<database>
[All facts and rules combined]
</database>

<query>
[The query only]
</query>

### CRITICAL FINAL REMINDERS (DO NOT IGNORE):
1. NO STANDARD ARITHMETIC ALLOWED: You MUST use CLP(FD) operators `#> `, `#>= `, `#< `, `#=< `, `#= ` for ALL math. `>` and `<` will instantly crash the parser!
2. CONTIGUOUS CLAUSES: Group ALL rules and facts that share the same predicate name together in the `<database>`. Do NOT scatter them.
---
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
---
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
lacks_magnetic_properties(rock_one).
lacks_magnetic_properties(rock_two).
far_apart(rock_one, rock_two).

-pulled_by_magnetism(RockA, _) :- lacks_magnetic_properties(RockA).
-pulled_by_magnetism(_, RockB) :- lacks_magnetic_properties(RockB).

pulled_together(ObjectA, ObjectB) :-
has_magnetic_properties(ObjectA),
has_magnetic_properties(ObjectB).
</database>

<query>
pulled_together(rock_one, rock_two).
</query>
---
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
Encode options.
choice(a, trench).
choice(b, rift_valley).
choice(c, ocean_basin).
choice(d, mid_ocean_ridge).
</reasoning_step>

<reasoning_step>
Encode formation logic explicitly from context.
forms_from(trench, subduction_zone).
forms_from(trench, plate_convergence).
forms_from(rift_valley, plate_divergence).
forms_from(ocean_basin, plate_divergence).
forms_from(mid_ocean_ridge, plate_divergence).
</reasoning_step>

<reasoning_step>
Define an `answer/1` rule that unifies with the correct option for the choice NOT associated with plate_divergence.
answer(OptionLetter) :- choice(OptionLetter, GeologicFeature), not forms_from(GeologicFeature, plate_divergence).
</reasoning_step>

<database>
choice(a, trench).
choice(b, rift_valley).
choice(c, ocean_basin).
choice(d, mid_ocean_ridge).

forms_from(trench, subduction_zone).
forms_from(trench, plate_convergence).
forms_from(rift_valley, plate_divergence).
forms_from(ocean_basin, plate_divergence).
forms_from(mid_ocean_ridge, plate_divergence).

answer(OptionLetter) :- choice(OptionLetter, GeologicFeature), not forms_from(GeologicFeature, plate_divergence).
</database>

<query>
answer(OptionLetter).
</query>
---
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
Encode options.
choice(a, dog).
choice(b, cat).
choice(c, human).
choice(d, frog).
</reasoning_step>

<reasoning_step>
Encode biological classifications explicitly from context.
is_mammal(dog).
is_mammal(cat).
is_mammal(human).
is_amphibian(frog).
</reasoning_step>

<reasoning_step>
Define an `answer/1` rule that unifies with the correct option for the choice NOT associated with being a mammal.
answer(OptionLetter) :- choice(OptionLetter, Animal), not is_mammal(Animal).
</reasoning_step>

<database>
choice(a, dog).
choice(b, cat).
choice(c, human).
choice(d, frog).

is_mammal(dog).
is_mammal(cat).
is_mammal(human).
is_amphibian(frog).

answer(OptionLetter) :- choice(OptionLetter, Animal), not is_mammal(Animal).
</database>

<query>
answer(OptionLetter).
</query>
---
"""},

    # Example 5: Scientific Reasoning MCQ (Domain-Specific)
    {
        "role": "user",
        "content": """
Context: Metals are good conductors of heat because they contain free-moving electrons that transfer kinetic energy quickly. Wood is a poor conductor of heat because its molecules are tightly bound and do not transfer energy efficiently. Rubber is an insulator. Glass conducts heat poorly.

Question: A chef needs to pick a material for a pot handle that stays cool while cooking. Which material would be the best choice?
A. copper
B. aluminum
C. wood
D. steel
        """
    },
    {"role": "assistant", "content": r"""
<reasoning_step>
Encode the four choices.
choice(a, copper).
choice(b, aluminum).
choice(c, wood).
choice(d, steel).
</reasoning_step>

<reasoning_step>
Encode conductivity from context.
is_conductor(copper).
is_conductor(aluminum).
is_conductor(steel).
is_insulator(wood).
</reasoning_step>

<reasoning_step>
The chef needs a handle that stays cool, meaning it should NOT conduct heat well.
A poor conductor (insulator) stays cool. From context, wood is a poor conductor.
stays_cool(Material) :- is_insulator(Material).
answer(OptionLetter) :- choice(OptionLetter, Material), stays_cool(Material).
</reasoning_step>

<database>
choice(a, copper).
choice(b, aluminum).
choice(c, wood).
choice(d, steel).

is_conductor(copper).
is_conductor(aluminum).
is_conductor(steel).
is_insulator(wood).

stays_cool(Material) :- is_insulator(Material).

answer(OptionLetter) :- choice(OptionLetter, Material), stays_cool(Material).
</database>

<query>
answer(OptionLetter).
</query>
---
"""},
]
EXPLAINER_SYSTEM_PROMPT = rf"""
### Role
You are a Logical Translator. Take the formal output of a Prolog s(CASP) solver and transform it into a clear, step-by-step natural language explanation in relation to the query and to arrive at an answer.

### Core Objectives
Narrate the Proof: Walk through the s(CASP) justification tree. Explain exactly why the conclusion was reached using the retrieved facts. If s(CASP) returns False or uses `not`, explain it as an "absence of evidence."
Keep your explanations dense and easy to synthesize and understand.

### Syntactic Rules
- No Prolog Syntax: Use natural language only.
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
        ] + GENERATOR_FEW_SHOT_EXAMPLES + [
            {'role': 'user', 'content': 'Acknowledge these strict Prolog generation instructions. You must extract a formal Prolog <database> and <query> strictly adhering to the s(CASP) syntactical constraints, with no standard arithmetic, no singletons, and contiguous clauses.'},
            {'role': 'assistant', 'content': 'I understand and will follow all instructions strictly. I will generate a syntactically valid s(CASP) Prolog program with a <reasoning_step>, <database>, and <query>. I will strictly avoid forbidden operators like `is`, `=\\=`, or `;`, and I will never use uppercase names for facts.'}
        ]

EXPLAINER_LLM_MESSAGES = [
            {'role': 'system', 'content': EXPLAINER_SYSTEM_PROMPT},
        ] + EXPLAINER_FEW_SHOT_EXAMPLES
