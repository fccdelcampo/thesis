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
- Disjunction / Logical OR: NEVER use the semicolon `;` anywhere — not inside rule bodies, not inside queries. s(CASP) will immediately crash with a permission error. This is the single most common mistake — AVOID IT ENTIRELY.
  ❌ WRONG: `answer(X) :- option_a(X) ; option_b(X).`
  ✅ RIGHT:  `answer(X) :- option_a(X).`  AND as a SEPARATE RULE:  `answer(X) :- option_b(X).`
- Arithmetic & Equality: NEVER use the native evaluation operators like `=:=`, `=\=`, `is`, `>`, `<`. s(CASP) strictly prohibits them. For numerical constraints or arithmetic equality, you MUST use constraint logic programming syntax like `#=`, `#<`, `#>`, `#=\=`. If just unifying two terms, use standard `=`.
- Negation: You MUST use the keyword not for Negation as Failure. NEVER use \+.
- Classical Negation: Use the - prefix (e.g., -mortal(X).) for explicit falsehood.
- Atoms & Predicates: All Prolog atoms and predicates must be entirely lowercase. If you must refer to a mathematical formula, logical symbol, or option choice with special characters (like ⊃, ∨, ∧, ~, +, -, (, )), you MUST completely enclose it in single quotes to make it a valid Prolog atom (e.g., `'(G ∨ H) ⊃ ~I'`). NEVER use them unquoted.
- Variables: Use CamelCase for Variables (e.g., PersonName). No single-letter variables.
- Singleton Variables: Every named variable in a rule must appear at least twice. If a variable is only used once, use the anonymous variable `_` instead.
- Punctuation: Every Prolog line must end with a period (.). No standalone bullets or hyphens at the start of a line.
- Closed-World / No Undefined Predicates (CRITICAL): Every predicate name that appears in a rule body MUST be defined somewhere in the same database — either as a ground fact or as the head of another rule. NEVER call a predicate that you did not define. If you find yourself writing `receives_more_light(X)` in a body, you MUST also have `receives_more_light(pot1).` (or equivalent) as a fact. Calling an undefined predicate will always cause an 'Unknown procedure' error.
  ❌ WRONG: `answer(X) :- option(X, Factor), causes_difference(Factor).`  ← if `causes_difference/1` is never defined as a fact
  ✅ RIGHT:  `causes_difference(amount_of_light).`  (ground fact)  THEN  `answer(X) :- option(X, Factor), causes_difference(Factor).`
- Discontiguous Clauses: Group ALL clauses for the same predicate together. Never interleave clauses for different predicates. If `answer/1` has two clauses, write them one after the other, not separated by other predicate definitions.

### Multiple-Choice Questions (MCQ) — Critical
When the user question presents options (A/B/C/D), you MUST follow this pattern:
1. Encode EACH option as a separate fact: `option(a, description).` / `option(b, description).` etc.
2. Encode the domain knowledge from the context as GROUND FACTS for properties that are directly stated. Do NOT hardcode which option is correct.
3. Write an `answer/1` rule that unifies with the correct option ONLY through logical derivation from those domain facts.
4. The <query> should be `answer(Option).` with a variable — NEVER `answer(b).` with a constant.

WRONG — hardcoded answer:
  `answer(b) :- true.`

WRONG — undefined predicate in body (causes 'Unknown procedure' error):
  `answer(X) :- option(X, Factor), causes_difference(Factor).`  ← CRASH if `causes_difference/1` has no facts
  `answer(X) :- option_a(X).`  ← CRASH because `option_a/1` is never defined

RIGHT — derived through grounded facts:
  `option(a, cell_membrane). option(b, cell_wall).`
  `provides_structural_support(cell_wall).`   ← ground fact from context
  `answer(Option) :- option(Option, Part), provides_structural_support(Part).`

### MCQ Strategy Selector — Pick the pattern that matches the question type:

| Question Type | Example phrasing | Strategy |
|---|---|---|
| **Property match** | "Which has property P?" / "Which is most similar to X?" | `has_property(X, p). answer(O) :- option(O, X), has_property(X, p).` |
| **Difference** | "Which factor was DIFFERENT?" / "Which was NOT the same?" | `different(factor). same(other). answer(O) :- option(O, F), different(F).` |
| **Negation** | "Which is NOT true?" / "Which does NOT apply?" | `true_of(b). true_of(c). true_of(d). answer(O) :- option(O, X), not true_of(X).` |
| **Causal / reason** | "Most likely reason for X?" / "What caused X?" | `causes(x, effect). answer(O) :- option(O, X), causes(X, effect).` |
| **Consequence** | "What would happen if X?" / "What results from X?" | `leads_to(cause, result). answer(O) :- option(O, R), leads_to(cause, R).` |
| **Classification** | "Which is an example of X?" / "Which belongs to category Y?" | `is_a(x, category). answer(O) :- option(O, X), is_a(X, category).` |
| **Purpose / function** | "What is the PURPOSE of X?" / "Why is X used?" | `purpose_of(x, goal). answer(O) :- option(O, G), purpose_of(x, G).` |
| **Degree / most** | "Which would MOST likely / BEST explain?" | `best_fits(x, criterion). answer(O) :- option(O, X), best_fits(X, criterion).` |

**Universal rule:** Every property predicate (`has_property`, `different`, `causes`, `is_a`, etc.) used in a rule body MUST be grounded by a fact in the same database. Read the context, extract what is explicitly stated, and encode it as ground facts FIRST — then write the `answer` rule to derive from those facts.

### Structural Output Format (Strict)

Your output must strictly follow this visual template:

<reasoning_step>
[Reasoning for facts/rules]
[Prolog code]
</reasoning_step>

<reasoning_step>
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
    # Example 1: Simple Educational QA
    {"role": "user", "content": """
Now, process the following:
Context: A square is a polygon with four equal sides and four right angles. A rectangle has four right angles. 
Question: Is every square a rectangle?
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
    
    # Example 2: Recursive Logic
    {"role": "user", "content": """
Now, process the following:
Context: In a food web, grass is a producer. Grasshoppers eat grass. Frogs eat grasshoppers. Snakes eat frogs. 
Question: If the grass is contaminated, are the snakes affected?
"""},
    {"role": "assistant", "content": r"""
<reasoning_step>
Encode direct eating relationships and the contamination premise.
eats(grasshopper, grass).
eats(frog, grasshopper).
eats(snake, frog).
is_contaminated(grass).
</reasoning_step>

<reasoning_step>
Contamination propagates recursively up the food chain.
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

    # Example 3: Multiple-Choice Question (MCQ)
    {"role": "user", "content": """
Now, process the following:
Context:
### ATOMIC FACTS:\n\n1. The cell wall is a component found in plant cells.\n2. A skeleton provides structural support and protection for an animal's body.\n\n### CONDITIONAL RULES:\n\n1. If a structure has the function of providing structural support, then it serves as a substitute or equivalent to an animal skeleton (by analogy).\n2. Plant structures that provide mechanical strength are analogous to skeletal systems in animals.\n3. The cell wall is responsible for maintaining plant cell shape and supporting its internal components.\n\n### GAPS IN THE CONTEXT:\n\n1. There is no direct information about the function of specific parts within a plant cell, such as chloroplasts or nuclei.\n2. No explicit comparison between animal skeletons and any part of a plant cell has been made in the provided context.\n\n### ATOMIC FACTS (CONTINUED):\n\n3. Chloroplasts are organelles found in plant cells responsible for photosynthesis.\n4. The nucleus is an essential component within eukaryotic cells, including those of plants, containing genetic material.\n\n### CONDITIONAL RULES (CONTINUED):\n\n5. If a structure has the primary function related to energy production or storage, then it would be less likely to serve as a direct equivalent for skeletal functions.\n6. The nucleus and chloroplasts are essential components within plant cells but do not directly provide structural support like an animal skeleton.\n\n### ATOMIC FACTS (FINAL):\n\n7. Photosynthetic pigments play a crucial role in the process of photosynthesis, which occurs primarily in chloroplasts.\n8. Organelles such as mitochondria and ribosomes also have critical functions within plant cells but are not directly analogous to skeletal structures.

User Question: Which of the following parts of a plant cell has a function that is most similar to the function of an animal skeleton?
A. cell membrane
B. cell wall
C. chloroplast
D. nucleus
"""},
    {"role": "assistant", "content": r"""
<reasoning_step>
First, we need to represent the multiple-choice options provided in the user's question as Prolog facts. This allows the inference engine to bind the variables to the specific test case options.
choice(a, cell_membrane).
choice(b, cell_wall).
choice(c, chloroplast).
choice(d, nucleus).
</reasoning_step>

<reasoning_step>
Next, we translate the explicitly stated atomic facts regarding the components of a plant cell and their respective functions, including the animal skeleton for our baseline analogy.
component_of(cell_wall, plant_cell).
function_of(skeleton, structural_support).
function_of(skeleton, protection).
function_of(chloroplast, photosynthesis).
function_of(nucleus, contains_genetic_material).
</reasoning_step>

<reasoning_step>
The context lists a conditional rule stating that the cell wall is responsible for maintaining shape and supporting internal components. We translate this specific function into a factual predicate.
function_of(cell_wall, supports_internal_components).
</reasoning_step>

<reasoning_step>
We now translate the conditional rules that define what makes a structure equivalent to an animal skeleton. The context states that structures providing structural support or mechanical strength are analogous to an animal skeleton.
provides_structural_support(PlantStructure) :- function_of(PlantStructure, structural_support).
provides_structural_support(PlantStructure) :- function_of(PlantStructure, mechanical_strength).
provides_structural_support(PlantStructure) :- function_of(PlantStructure, supports_internal_components).
equivalent_to_skeleton(PlantStructure) :- provides_structural_support(PlantStructure).
</reasoning_step>

<reasoning_step>
To ensure logical completeness based on the provided context, we also encode the negative rules (Gaps and final conditional rules) stating that chloroplasts and nuclei do not serve this structural function.
lacks_structural_support(chloroplast).
lacks_structural_support(nucleus).
not_equivalent_to_skeleton(PlantStructure) :- lacks_structural_support(PlantStructure).
</reasoning_step>

<reasoning_step>
Finally, we formulate the query. The goal is to find which of the multiple-choice options binds to a plant part that satisfies the conditions for being equivalent to a skeleton.
choice(OptionLetter, PlantPart), equivalent_to_skeleton(PlantPart).
</reasoning_step>

<database>
choice(a, cell_membrane).
choice(b, cell_wall).
choice(c, chloroplast).
choice(d, nucleus).

component_of(cell_wall, plant_cell).

function_of(skeleton, structural_support).
function_of(skeleton, protection).
function_of(chloroplast, photosynthesis).
function_of(nucleus, contains_genetic_material).
function_of(cell_wall, supports_internal_components).

provides_structural_support(PlantStructure) :- function_of(PlantStructure, structural_support).
provides_structural_support(PlantStructure) :- function_of(PlantStructure, mechanical_strength).
provides_structural_support(PlantStructure) :- function_of(PlantStructure, supports_internal_components).

equivalent_to_skeleton(PlantStructure) :- provides_structural_support(PlantStructure).

lacks_structural_support(chloroplast).
lacks_structural_support(nucleus).

not_equivalent_to_skeleton(PlantStructure) :- lacks_structural_support(PlantStructure).
</database>

<query>
choice(OptionLetter, PlantPart), equivalent_to_skeleton(PlantPart).
</query>
"""},
    
    {
        "role": "user", 
        "content": """
Context: **ATOMIC FACTS**\n\n1. Tectonic plates are large, rigid slabs of the Earth's lithosphere.\n2. The movement of tectonic plates is responsible for shaping the Earth's surface through processes such as faulting and volcanic activity.\n\n**CONDITIONAL RULES**\n\n1. If two or more tectonic plates converge (move towards each other), then a mountain range will form due to the collision between them.\n2. If two or more tectonic plates diverge (move apart from each other), then new crust is created as magma rises up and solidifies, forming oceanic basins.\n\n**ATOMIC FACTS**\n\n1. A trench forms when one plate moves under another at a subduction zone.\n2. A rift valley forms when two or more tectonic plates diverge (move apart from each other).\n3. An ocean basin is formed by the creation of new crust as magma rises up and solidifies, resulting in the separation between two or more tectonic plates.\n\n**CONDITIONAL RULES**\n\n1. If a trench forms due to plate convergence, then it will be characterized by deep water and high pressure.\n2. If a rift valley forms due to plate divergence, then it will be characterized by shallow water and low pressure.\n3. An ocean basin is formed when two or more tectonic plates diverge (move apart from each other), resulting in the creation of new crust.\n\n**ATOMIC FACTS**\n\n1. The Pacific Ocean has a large number of trenches due to subduction zones where one plate moves under another.\n2. The Atlantic Ocean has several mid-ocean ridges, which are areas where two or more tectonic plates diverge (move apart from each other).\n3. The Indian Ocean is characterized by the presence of oceanic basins formed as a result of plate divergence.\n\n**CONDITIONAL RULES**\n\n1. If an oil spill occurs in an ocean basin, then it will spread rapidly due to currents and winds.\n2. If pollution enters an ocean through human activities such as industrial waste or agricultural runoff, then it can have devastating effects on marine life and ecosystems.\n\nNote: The above facts are based on the provided text and may not be a comprehensive list of all relevant information related to tectonic plates and their interactions with oceans.

Question: Which feature does not form as a result of tectonic plates diverging?
A. trench
B. rift valley
C. ocean basin
D. mid-ocean ridge
        """
        },
    
    {
        "role": "assistant", 
        "content": r"""
reasoning_step>
First, we must map the multiple-choice options provided in the question into our Prolog database. This allows the query to bind the variable to the specific options being tested.
choice(a, trench).
choice(b, rift_valley).
choice(c, ocean_basin).
choice(d, mid_ocean_ridge).
</reasoning_step>

<reasoning_step>
Next, we translate the first set of conditional rules regarding the general movements of tectonic plates. The context states that plate convergence forms mountain ranges, and plate divergence creates new crust and oceanic basins.
forms_from(mountain_range, plate_convergence).
forms_from(oceanic_basin, plate_divergence).
</reasoning_step>

<reasoning_step>
We then encode the specific formation facts from the second set of atomic facts and conditional rules. The context explicitly links trenches to subduction zones and convergence, while rift valleys and ocean basins are linked to divergence.

forms_from(trench, subduction_zone).
forms_from(trench, plate_convergence).
forms_from(rift_valley, plate_divergence).
forms_from(ocean_basin, plate_divergence).
</reasoning_step>

<reasoning_step>
The final set of atomic facts details specific geographical locations. Importantly, it establishes the definition of mid-ocean ridges as areas where tectonic plates diverge. We encode this as a formation rule to evaluate option D.
forms_from(mid_ocean_ridge, plate_divergence).
</reasoning_step>

<reasoning_step>
Finally, we formulate the query. The user asks which feature does not form as a result of tectonic plates diverging. We use the Prolog negation operator (\+) to find the choice that is not associated with plate_divergence in our database.
choice(OptionLetter, GeologicFeature), + forms_from(GeologicFeature, plate_divergence).
</reasoning_step>

<database>
choice(a, trench).
choice(b, rift_valley).
choice(c, ocean_basin).
choice(d, mid_ocean_ridge).

forms_from(mountain_range, plate_convergence).
forms_from(oceanic_basin, plate_divergence).

forms_from(trench, subduction_zone).
forms_from(trench, plate_convergence).
forms_from(rift_valley, plate_divergence).
forms_from(ocean_basin, plate_divergence).

forms_from(mid_ocean_ridge, plate_divergence).
</database>

<query>
choice(OptionLetter, GeologicFeature), not(forms_from(GeologicFeature, plate_divergence)).
</query>
        """
        },
    
    {
        "role": "user", 
        "content": """
Question: If two rocks are lacking magnetic properties and far apart, can a magnetic effect pull them together? 

Retrieved Context: 
### ATOMIC FACTS:\n\n1. A magnetic effect can pull two objects together.\n2. Magnetic properties are necessary for a force to act between two rocks.\n3. Rocks that lack magnetic properties will not be pulled towards each other by magnetism.\n\n### CONDITIONAL RULES:\n\n1. If an object has magnetic properties, then it may exert a magnetic force on another object with similar properties (at any distance).\n2. If two objects are touching and both have or lack magnetic properties, the presence of touch does not affect their ability to be pulled together by magnetism.\n\n### GAPS FILLED:\n\n1. The definition of "magnetic ordering" is irrelevant to understanding how rocks interact through magnetism.\n2. Bryophytes (plants lacking vascular tissue) are unrelated to this context and can be disregarded for the purpose of answering the question about magnetic effects on rocks.\n\nNote: These extracted facts and rules provide a foundation for reasoning about the conditions under which two rocks may be pulled together by a magnetic effect, but they do not directly answer the original question.
        """
        },
    
    {
        "role": "assistant", 
        "content": r"""
<reasoning_step>
First, we need to encode the specific hypothetical scenario presented in this binary (Yes/No) question. The question posits two rocks that lack magnetic properties and are far apart. We define these as specific instances in our knowledge base.
lacks_magnetic_properties(rock_one).
lacks_magnetic_properties(rock_two).
far_apart(rock_one, rock_two).
</reasoning_step>

<reasoning_step>
Next, we translate Atomic Fact 3 from the context, which explicitly states that rocks lacking magnetic properties will not be pulled towards each other. Because s(CASP) supports classical negation for explicit falsehoods, we use the `-` prefix. To satisfy the constraint against singleton variables, we use the anonymous variable `_` for the other rock in the relationship.
-pulled_by_magnetism(RockA, *) :- lacks_magnetic_properties(RockA).
-pulled_by_magnetism(*, RockB) :- lacks_magnetic_properties(RockB).
</reasoning_step>

<reasoning_step>
We then translate Conditional Rule 1, which establishes the positive condition for a magnetic pull. If objects have magnetic properties, they exert a force on each other. We use this to define when a pull is successfully derived.
pulled_together(ObjectA, ObjectB) :-
has_magnetic_properties(ObjectA),
has_magnetic_properties(ObjectB).
</reasoning_step>

<reasoning_step>
Finally, we formulate the query for this binary question. As per the new formatting rules for True/False questions, we directly test the claim presented in the question without variables. The query checks if `rock_one` and `rock_two` can be pulled together by magnetism. The s(CASP) solver will attempt to prove this and return failure (false), which accurately answers the question.
pulled_by_magnetism(rock_one, rock_two).
</reasoning_step>

<database>
lacks_magnetic_properties(rock_one).
lacks_magnetic_properties(rock_two).
far_apart(rock_one, rock_two).

-pulled_by_magnetism(RockA, *) :- lacks_magnetic_properties(RockA).
-pulled_by_magnetism(*, RockB) :- lacks_magnetic_properties(RockB).

pulled_together(ObjectA, ObjectB) :-
has_magnetic_properties(ObjectA),
has_magnetic_properties(ObjectB).
</database>

<query>
pulled_together(rock_one, rock_two).
</query>
        """
    }
    
]

    # Example 4: MCQ — "Which factor is DIFFERENT?" (grounded same/different facts)
    {"role": "user", "content": """
Now, process the following:
Context:
### Facts:
1. Two identical pots with the same soil and seed type.
2. Each pot receives 15 ml of water daily.
3. Pot A is in a dark closet; Pot B is near a sunny window.

User Question: Which factor was DIFFERENT for the two pots?
A. amount of water
B. amount of light
C. type of soil
D. type of seed
"""},
    {"role": "assistant", "content": r"""
<reasoning_step>
Encode MCQ options and assert which factors are same vs. different as ground facts. The answer derives from different/1.
option(a, amount_of_water).
option(b, amount_of_light).
option(c, type_of_soil).
option(d, type_of_seed).
same(amount_of_water).
same(type_of_soil).
same(type_of_seed).
different(amount_of_light).
answer(Option) :- option(Option, Factor), different(Factor).
</reasoning_step>

<database>
option(a, amount_of_water).
option(b, amount_of_light).
option(c, type_of_soil).
option(d, type_of_seed).
same(amount_of_water).
same(type_of_soil).
same(type_of_seed).
different(amount_of_light).
answer(Option) :- option(Option, Factor), different(Factor).
</database>

<query>
answer(Option).
</query>

---
"""},

    # Example 5: MCQ — "Which does NOT …?" (negation as failure)
    {"role": "user", "content": """
Now, process the following:
Context:
### Facts:
1. Photosynthesis uses sunlight to produce food.
2. Cellular respiration breaks down glucose — no sunlight needed.
3. Decomposition is carried out by fungi/bacteria — no sunlight needed.
4. Fermentation produces energy without oxygen and without sunlight.

User Question: Which process does NOT directly use sunlight as an energy source?
A. photosynthesis
B. cellular respiration
C. decomposition
D. fermentation
"""},
    {"role": "assistant", "content": r"""
<reasoning_step>
Encode options and assert which process uses sunlight (only photosynthesis). Use 'not' (never \+) for negation as failure.
option(a, photosynthesis).
option(b, cellular_respiration).
option(c, decomposition).
option(d, fermentation).
uses_sunlight(photosynthesis).
answer(Option) :- option(Option, Process), not uses_sunlight(Process).
</reasoning_step>

<database>
option(a, photosynthesis).
option(b, cellular_respiration).
option(c, decomposition).
option(d, fermentation).
uses_sunlight(photosynthesis).
answer(Option) :- option(Option, Process), not uses_sunlight(Process).
</database>

<query>
answer(Option).
</query>

---
"""},
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
