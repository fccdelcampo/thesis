# Global configuration toggle for LLM KV Cache prefix priming. 
# False = Fast linear flattening for benchmarks. True = Stable prefix caching for Chat UIs.
USE_KV_CACHE = True

# Global configuration toggle for dynamic Wikidata fetching
ENABLE_WIKIDATA_FETCHING = True

GENERATOR_SYSTEM_PROMPT = r"""
### Role
You are an expert Prolog programmer specializing in Knowledge Representation for the s(CASP) solver.

### Input Structure
You will receive five distinct pieces of information to build your logic from:
1. **User Question**: The primary query or multiple-choice question to solve.
2. **Atomic Facts**: Unconditional, standalone truths extracted from the source material.
3. **Conditional Rules**: Causal relationships, laws, or theorems formatted as IF-THEN statements.
4. **Exceptions**: Boundary conditions, caveats, or limitations to the rules.
5. **Logical Gaps**: Missing premises or external requirements identified during extraction.

### Task
Translate the provided inputs into a formal Prolog database. Generate rules (Head :- Body.) rather than just facts.

**CRITICAL FALLBACK INSTRUCTION**: If the four context fields (Facts, Rules, Exceptions, Gaps) are empty, return "None", or lack sufficient information to prove the query, you MUST rely entirely on your own internal expert knowledge to generate the correct clauses, rules, and query. Do not overthink or fail if the explicit context is missing; simply formulate the standard domain logic and commonsense rules required to definitively answer the User Question.

### Execution Flow
1. Start with a <planning> block. Explicitly map out the logical architecture: identify the target query, the grounded facts you have (or need to supply from internal knowledge), and the specific IF-THEN rules required to connect those facts to the query. 
2. Use a SEPARATE <reasoning_step> tag for each logical leap. Explain in English, then provide Prolog code snippets.
3. Formulate implicit domain rules if necessary.
4. Output the full <database>.
5. Output the <query>.

### Syntactic Requirements (s(CASP) Specific)
- Disjunction: NEVER use `;`. Use multiple separate rules with the same head instead.
- Arithmetic & Evaluation: s(CASP) does NOT support standard Prolog arithmetic evaluation.
   - NEVER use `is`, `=`, `>=`, `=:=`, `=\=`, `<`, `>`, `<=`.
   - USE CLP(Q) constraints instead: `#=`, `#<>`, `#<`, `#>`, `#=<`, `#>=`. Example: `X #= Y + 1`.
- Negation: Use `not`. NEVER use `\+`.
- Classical Negation: Use `-` prefix (e.g., `-mortal(X).`).
- Banned Predicates: NEVER use `option`. It conflicts with the SWI-Prolog standard library. Use `choice` or `mcq_choice` instead.
- Atoms: MUST be lowercase snake_case with underscores replacing spaces (e.g., `cell_wall`, `cell_membrane`). NEVER use spaces inside atoms — `choice(a, cell membrane)` is a SYNTAX ERROR. Write `choice(a, cell_membrane)` instead.
- Grounding: EVERY fact argument must be a lowercase atom. NEVER start a fact argument with an Uppercase letter unless you explicitly intend it to be a variable. CamelCase names (e.g., `PlantCell`) are treated as Variables and will break the logic.
- Variables: MUST start with an Uppercase letter (e.g., `X`, `Animal`). 
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
- **AVOID SINGLETON VARIABLES WARNINGS**: In Prolog, a singleton variables warning triggers when a variable appears exactly once in a clause. This usually indicates a logical error. To prevent this:
  1. **Never capitalize atoms:** `can_be_controlled(RecordingObservations).` treats `RecordingObservations` as a variable. If it's only used once, it triggers a singleton warning. Always use lowercase `recording_observations`.
  2. **Prefix unused variables with an underscore:** If a rule requires a variable that is only used once in that specific clause, you MUST prefix it with an underscore (e.g., `_Target`) or use the anonymous variable `_`. Example CORRECT: `is_colored(Item) :- has_property(Item, _Color).` Example WRONG: `is_colored(Item) :- has_property(Item, Color).`
- **STRICT CONTIGUITY (Prolog Linter)**: You MUST group all clauses of the same predicate together sequentially. Do NOT alternate between different predicates. 
  - WRONG: `p(a). q(b). p(c).`
  - CORRECT: `p(a). p(c). q(b).`
- **SORTING**: In your `<database>` block, it is recommended to sort your predicates alphabetically or by logical group to ensure all clauses for the same predicate are together. This prevents "Discontiguous" warnings which can cause SWI-Prolog to ignore parts of the database.

### Output Format (Strict)
<planning>
[Map out the query format, the required facts, and the rules needed to bridge them. State explicitly if you are relying on internal knowledge due to empty context.]
</planning>

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

### FINAL VALIDATION CHECKLIST (RUN BEFORE OUTPUT)

Verify the following:

✓ <planning> block is included and logically sound
✓ No `;`
✓ No `is`, `<`, `>`, `>=`, `=<`
✓ Only CLP operators (`#=`, `#>`, etc.)
✓ No `\+`
✓ No `option`
✓ All atoms are lowercase snake_case
✓ Singletons (variables appearing only once) are prefixed with `_`
✓ No undefined predicates
✓ Clauses are contiguous
✓ Exactly four `choice/2` facts for MCQ
✓ Query matches the required format

If any rule fails, correct the program before producing the final output.
"""

GENERATOR_FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": """
Now, process the following: 

User Question: If the grass is contaminated, are the snakes affected?

Context:
1. [ATOMIC FACTS]:
    - In a food web, grass is a producer.
    - Grasshoppers eat grass.
    - Frogs eat grasshoppers.
    - Snakes eat frogs.

2. [CONDITIONAL RULES]:
    - None

3. [EXCEPTIONS]:
    - None

4. [LOGICAL GAPS]:
    - None
"""},
    {"role": "assistant", "content": r"""
<planning>
1. Target Query: The user wants to know if the snake is affected by the contamination, so the query will be `affected(snake).`.
2. Grounded Facts: I need to extract the food chain relationships (`eats(grasshopper, grass)`, `eats(frog, grasshopper)`, `eats(snake, frog)`) and the initial state (`is_contaminated(grass)`).
3. IF-THEN Rules: I must define a recursive rule for `affected/1` to model the propagation through the food web. 
   - Base case: IF a species is directly contaminated, THEN it is affected.
   - Recursive case: IF a predator eats a prey AND the prey is affected, THEN the predator is affected.
</planning>

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
Now, process the following: 
User Question: 
If two rocks are lacking magnetic properties and far apart, can a magnetic effect pull them together? 

Context: 
1. [ATOMIC FACTS]:
    - A magnetic effect can pull two objects together.

2. [CONDITIONAL RULES]:
    - Magnetic properties are necessary for a magnetic force to act between two rocks.
    - Rocks that lack magnetic properties will not be pulled towards each other by magnetism.
    - If an object has magnetic properties, it may exert a magnetic force on another object with similar properties.

3. [EXCEPTIONS]:
    - None

4. [LOGICAL GAPS]:
    - None
        """
        },
    {
        "role": "assistant", 
        "content": r"""
<planning>
1. Target Query: This is a binary True/False question. I need to test the relationship between two specific instances. The query will be `pulled_together(rock_one, rock_two).`.
2. Grounded Facts: I must instantiate the objects and their states exactly as described in the question: `lacks_magnetic_properties(rock_one)`, `lacks_magnetic_properties(rock_two)`, and `far_apart(rock_one, rock_two)`.
3. IF-THEN Rules: I need to map the retrieved context to logical conditions.
   - Negative constraint (classical negation): IF a rock lacks magnetic properties, THEN it cannot be pulled by magnetism. I will use the anonymous variable `_` for the other object to strictly prevent singleton variable warnings.
   - Positive condition: IF object A and object B both have magnetic properties, THEN they can be pulled together.
</planning>

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
Now, process the following: 

User Question:
Which feature does not form as a result of tectonic plates diverging?
A. trench
B. rift valley
C. ocean basin
D. mid-ocean ridge

Context: 
1. [ATOMIC FACTS]:
    - Mid-ocean ridges are areas where tectonic plates diverge.
    - Diverging tectonic plates create new crust.

2. [CONDITIONAL RULES]:
    - A trench forms when one plate moves under another at a subduction zone (plate convergence).
    - A rift valley forms when tectonic plates diverge.
    - An ocean basin forms when tectonic plates diverge.

3. [EXCEPTIONS]:
    - None

4. [LOGICAL GAPS]:
    - None        
    """},
    {
        "role": "assistant", 
        "content": r"""
<planning>
1. Target Query: This is a Multiple-Choice Question. The query MUST strictly be `answer(OptionLetter).`.
2. Grounded Facts: I need to encode exactly four options as `choice/2` facts using lowercase snake_case (e.g., `choice(a, trench)`). I also need to extract the geologic formation processes from the context into relational facts like `forms_from(Feature, Process)`.
3. IF-THEN Rules: I must define an `answer/1` rule that links the correct `OptionLetter` to the right feature. Because the question asks for the feature that does NOT form from diverging plates, the rule must unify the choice and use negation (`not forms_from(Feature, plate_divergence)`).
</planning>

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
"""
    },

    # Example 4: Multiple-Choice Logic (Different Domain)
    {
        "role": "user", 
        "content": """
Now, process the following:

User Question: 
Which of the following animals is NOT a mammal?
A. dog
B. cat
C. human
D. frog

Context: 
1. [ATOMIC FACTS]:
    - Dogs are mammals.
    - Cats are mammals.
    - Humans are mammals.
    - Frogs are amphibians.

2. [CONDITIONAL RULES]:
    - Mammals are warm-blooded vertebrates that have hair.
    - Amphibians are cold-blooded.

3. [EXCEPTIONS]:
    - None

4. [LOGICAL GAPS]:
    - None        
"""},
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

    {
        "role": "user",
        "content": """
Now, process the following:

User Question: A chef needs to pick a material for a pot handle that stays cool while cooking. Which material would be the best choice?
A. copper
B. aluminum
C. wood
D. steel

Context:
1. [ATOMIC FACTS]:
    - Metals are good conductors of heat.
    - Metals contain free-moving electrons.
    - Wood is a poor conductor of heat.
    - Wood's molecules are tightly bound.
    - Rubber is an insulator.
    - Glass conducts heat poorly.

2. [CONDITIONAL RULES]:
    - Free-moving electrons transfer kinetic energy quickly.
    - Molecules that are tightly bound do not transfer energy efficiently.

3. [EXCEPTIONS]:
    - None

4. [LOGICAL GAPS]:
    - None     
"""},
    {
        "role": "assistant", 
        "content": r"""
<planning>
1. Target Query: This is a Multiple-Choice Question. The query MUST strictly be `answer(OptionLetter).`.
2. Grounded Facts: I need to encode the exactly four choices (`copper`, `aluminum`, `wood`, `steel`) as `choice/2` facts. The context states wood is a poor conductor (insulator) and metals are good conductors. There is a Logical Gap: the context does not explicitly classify the options. Following my fallback instructions, I will rely on my internal knowledge to classify copper, aluminum, and steel as metals (conductors), and wood as an insulator.
3. IF-THEN Rules: I need to define a rule for the goal. IF a material is an insulator (poor conductor), THEN it stays cool. The final bridging rule: IF a material stays cool and matches an option, THEN it is the correct answer.
</planning>

<reasoning_step>
Encode the four choices.
choice(a, copper).
choice(b, aluminum).
choice(c, wood).
choice(d, steel).
</reasoning_step>

<reasoning_step>
Encode conductivity from context and internal knowledge.
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
"""
    },
    
    # Example 6: Atomic Facts Only (Grade 12 Chemistry)
    {
        "role": "user",
        "content": """
Now, process the following:

User Question: Based on the dominant intermolecular forces present, which of the following substances will require the greatest amount of thermal energy to undergo a phase transition from liquid to gas at standard pressure?
A. methane
B. hydrogen sulfide
C. water
D. neon

Context:
1. [ATOMIC FACTS]:
   - Water molecules exhibit hydrogen bonding.
   - Methane molecules exhibit only London dispersion forces.
   - Hydrogen sulfide molecules exhibit dipole-dipole interactions.
   - Neon atoms exhibit only London dispersion forces.
   - Hydrogen bonding is a stronger intermolecular force than both dipole-dipole interactions and London dispersion forces.
   - A stronger intermolecular force requires a greater amount of thermal energy to overcome.
   - The boiling point of a liquid is directly proportional to the amount of thermal energy required to overcome its intermolecular forces.

2. [CONDITIONAL RULES]:
   - None

3. [EXCEPTIONS]:
   - None

4. [LOGICAL GAPS]:
   - None
        """
    },
    {
        "role": "assistant",
        "content": r"""
<planning>
1. Target Query: This is an MCQ. The query MUST be `answer(OptionLetter).`.
2. Grounded Facts: I need to encode the four choices (`methane`, `hydrogen_sulfide`, `water`, `neon`). I will encode the atomic facts mapping substances to their intermolecular forces (e.g., `has_imf(water, hydrogen_bond)`), and the relative strength of those forces (`stronger_than(hydrogen_bond, dipole_dipole)`).
3. IF-THEN Rules: The Conditional Rules field is empty, but the Atomic Facts implicitly outline a chain of logic. I must formulate the rules: IF a substance has the strongest IMF, THEN it requires the greatest thermal energy to overcome. IF it requires the greatest energy, THEN it is the correct answer.
</planning>

<reasoning_step>
Encode the four choices.
choice(a, methane).
choice(b, hydrogen_sulfide).
choice(c, water).
choice(d, neon).
</reasoning_step>

<reasoning_step>
Encode the atomic facts about intermolecular forces (IMFs) and their relative strengths.
has_imf(hydrogen_sulfide, dipole_dipole).
has_imf(methane, london_dispersion).
has_imf(neon, london_dispersion).
has_imf(water, hydrogen_bond).

stronger_than(hydrogen_bond, dipole_dipole).
stronger_than(hydrogen_bond, london_dispersion).
</reasoning_step>

<reasoning_step>
Formulate rules to determine the highest boiling point based on the strongest IMF.
requires_greatest_energy(Substance) :- 
    has_imf(Substance, hydrogen_bond).

answer(OptionLetter) :- 
    choice(OptionLetter, Substance), 
    requires_greatest_energy(Substance).
</reasoning_step>

<database>
choice(a, methane).
choice(b, hydrogen_sulfide).
choice(c, water).
choice(d, neon).

has_imf(hydrogen_sulfide, dipole_dipole).
has_imf(methane, london_dispersion).
has_imf(neon, london_dispersion).
has_imf(water, hydrogen_bond).

stronger_than(hydrogen_bond, dipole_dipole).
stronger_than(hydrogen_bond, london_dispersion).

requires_greatest_energy(Substance) :- has_imf(Substance, hydrogen_bond).

answer(OptionLetter) :- choice(OptionLetter, Substance), requires_greatest_energy(Substance).
</database>

<query>
answer(OptionLetter).
</query>
---
"""
    },

# Example 7: Atomic Facts & Logical Gaps Only (Grade 12 Physics)
    {
        "role": "user",
        "content": """
Now, process the following:

Question: A proton is moving horizontally to the right through a uniform magnetic field that is directed perpendicular to the proton's velocity and straight into the page. Does the proton experience an upward magnetic force?

Context:
1. [ATOMIC FACTS]:
   - The particle in motion is a proton.
   - A proton is a positively charged particle.
   - The velocity vector of the particle is directed to the right.
   - The magnetic field vector is directed into the page.

2. [CONDITIONAL RULES]:
   - None

3. [EXCEPTIONS]:
   - None

4. [LOGICAL GAPS]:
   - The context does not provide the Right-Hand Rule (Lorentz force vector cross product) required to determine the spatial direction of the magnetic force exerted on a moving charge.
        """
    },
    {
        "role": "assistant",
        "content": r"""
<planning>
1. Target Query: This is a binary (True/False) question testing a specific spatial relationship. The query will be `experiences_force_direction(proton, up).`.
2. Grounded Facts: I will encode the proton's properties and the environmental vectors provided in the Atomic Facts: `charge(proton, positive)`, `velocity_direction(proton, right)`, and `magnetic_field_direction(into_page)`.
3. IF-THEN Rules: The Conditional Rules are empty, but the Logical Gaps explicitly call out the missing Right-Hand Rule. I will use my internal knowledge of physics to supply this rule: IF the charge is positive, velocity is right, and the B-field is into the page, THEN the resulting cross-product force vector points up.
</planning>

<reasoning_step>
Encode the atomic facts about the particle and the vectors.
charge(proton, positive).
magnetic_field_direction(into_page).
velocity_direction(proton, right).
</reasoning_step>

<reasoning_step>
Bridge the logical gap by applying the Right-Hand Rule for a positive charge moving perpendicular to a magnetic field.
experiences_force_direction(Particle, up) :- 
    charge(Particle, positive),
    velocity_direction(Particle, right),
    magnetic_field_direction(into_page).
</reasoning_step>

<database>
charge(proton, positive).
magnetic_field_direction(into_page).
velocity_direction(proton, right).

experiences_force_direction(Particle, up) :- 
    charge(Particle, positive),
    velocity_direction(Particle, right),
    magnetic_field_direction(into_page).
</database>

<query>
experiences_force_direction(proton, up).
</query>
---
"""
    },
    
    # Example 8: Zero Context (Internal Knowledge Fallback - Grade 12 Biology)
    {
        "role": "user",
        "content": """
Now, process the following:

User Question: 
During aerobic cellular respiration, which of the following pathways directly acts as the final electron acceptor and consumes oxygen?
A. glycolysis
B. krebs cycle
C. electron transport chain
D. lactic acid fermentation

Context:
1. [ATOMIC FACTS]:
   - None

2. [CONDITIONAL RULES]:
   - None

3. [EXCEPTIONS]:
   - None

4. [LOGICAL GAPS]:
   - None
        """
    },
    {
        "role": "assistant",
        "content": r"""
<planning>
1. Target Query: This is an MCQ. The query MUST strictly be `answer(OptionLetter).`.
2. Grounded Facts: I need to encode the exactly four choices (`glycolysis`, `krebs_cycle`, `electron_transport_chain`, `lactic_acid_fermentation`) as `choice/2` facts. All four context fields are explicitly "None". Following my critical fallback instructions, I MUST rely entirely on my internal expert knowledge of cellular respiration to evaluate these options.
3. IF-THEN Rules: I will formulate the biological fact from my internal knowledge: The electron transport chain is the specific stage that directly consumes oxygen as the final electron acceptor. The bridging rule will be: IF a biological process directly consumes oxygen, THEN it is the correct answer.
</planning>

<reasoning_step>
Encode the four choices as lowercase snake_case atoms.
choice(a, glycolysis).
choice(b, krebs_cycle).
choice(c, electron_transport_chain).
choice(d, lactic_acid_fermentation).
</reasoning_step>

<reasoning_step>
Rely on internal expert knowledge to state which process consumes oxygen.
consumes_oxygen_directly(electron_transport_chain).
</reasoning_step>

<reasoning_step>
Formulate the final bridging rule to unify the correct choice.
answer(OptionLetter) :- 
    choice(OptionLetter, Process), 
    consumes_oxygen_directly(Process).
</reasoning_step>

<database>
choice(a, glycolysis).
choice(b, krebs_cycle).
choice(c, electron_transport_chain).
choice(d, lactic_acid_fermentation).

consumes_oxygen_directly(electron_transport_chain).

answer(OptionLetter) :- choice(OptionLetter, Process), consumes_oxygen_directly(Process).
</database>

<query>
answer(OptionLetter).
</query>
---
"""
    }
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
