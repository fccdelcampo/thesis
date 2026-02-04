from pathlib import Path

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

PROMPT_TEMPLATE = '''
You are an educational question answering model tasked with extracting information from papers
and structuring it in a property graph to inform students about their potential academic questions.
EXTRACT JSON ONLY. DO NOT TALK. NO PREAMBLE. NO MARKDOWN.
Do not generate redundant relationships. Create only ONE relationship between any two nodes. Prioritize 'DEFINES' or 'PART_OF' over 'RULE' or 'EXCEPTION' unless strictly necessary.

Extract the entities (nodes) and specify their type from the following Input text.
Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node.


STRICTLY return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
  "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

...

Use only fhe following nodes and relationships:
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and the relationship direction.

Do not return any additional information other than the JSON in it.


Examples:
{examples}

Input text:

{text}

AGAIN EXTRACT JSON ONLY. DO NOT TALK. NO PREAMBLE. NO MARKDOWN.
'''

PROMPT_TEMPLATE = '''
### Role
You are a Knowledge Engineer specialized in Symbolic Logic. 
Your goal is to extract evidence from text to build a Prolog-compatible Knowledge Graph.
</system>

### Task
Decompose the input text into entities and relationships. 
For every relationship, you must provide a 'natural_statement' that preserves the full 
nuance of the source text (modality, quantifiers, conditions).

### Execution Flow
IDENTIFY THREE TYPES OF LOGIC:
1. FACT: A simple assertion (e.g., "X is a Y").
2. RULE: A conditional implication (e.g., "If P then Q").
3. EXCEPTION: A negation or constraint (e.g., "Unless Z, then P").

### STRICT JSON FORMAT:
{{
  "nodes": [
    {{
      "id": "unique_atom_id", 
      "label": "PrologSafeLabel", 
      "properties": {{"name": "Actual Name"}} 
    }}
  ],
  "relationships": [
    {{
      "type": "PREDICATE_TYPE",
      "start_node_id": "id1",
      "end_node_id": "id2",
      "properties": {{
        "natural_statement": "The full natural language fact or rule extracted from text",
        "logic_intent": "FACT | RULE | EXCEPTION",
        "logical_form_hint": "e.g., parent(X, Y) or mortal(X) :- man(X)"
      }}
    }}
  ]
}}
</task>

### Constraints
- Use lowercase_with_underscores for IDs and Types (Prolog atom style).
- Do not hallucinate. If no clear logic exists, return empty arrays.
- {schema}
</constraints>

### Examples
{examples}

### Input Text
{text}

FINAL INSTRUCTION: Output the extracted data as a JSON object ONLY. No markdown, no code blocks, no preamble.
'''

RETRIEVAL_QUERY = '''
"""
MATCH (node)-[:EXTRACTS]->(e:Entity)
OPTIONAL MATCH (e)-[r]->(target:Entity)
RETURN node.text AS text, 
       node.score AS score,
       {entity: e.name, relationship: type(r), connected_to: target.name} AS graph_metadata
"""
'''
GRAPHRAG_TEMPLATE = '''
Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned.

# Question:
{query_text}

# Context:
{context}

# Answer:
'''

# GRAPHRAG_TEMPLATE = '''
# ### ROLE
# You are a Logic Discovery Agent. Your task is NOT to answer the user's question. 
# Instead, you must extract all individual pieces of evidence from the provided Context 
# that are necessary to answer the Question.

# ### TASK
# Analyze the Context and the Question. Extract the following information in a clear, 
# bulleted list format. If a category has no relevant info, write "None".

# ### CATEGORIES OF EVIDENCE TO EXTRACT:
# 1. **ATOMIC FACTS**: Simple, unconditional truths (e.g., "Socrates is a man").
# 2. **CONDITIONAL RULES**: Implications or 'If-Then' relationships (e.g., "If X is a man, then X is mortal").
# 3. **LOGICAL EXCEPTIONS**: Constraints, negations, or 'Unless' cases (e.g., "Penguins are birds but cannot fly").
# 4. **NUMERIC/MODAL DATA**: Any specific quantities, dates, or degrees of certainty (e.g., "75% probability").

# ### STRICT RULES:
# - DO NOT answer the question.
# - DO NOT use conversational filler.
# - ONLY include information found in the Context.
# - Use precise language that can be mapped to Prolog atoms.
# </task>

# ### Question:
# {query_text}

# ### Context:
# {context}

# '''

DOC_PATH = Path(__file__).resolve().parents[2] / "documents"

ENCODER_SYSTEM_PROMPT = '''
### ROLE
You are a Linguistic Pre-processor for a Formal Logic System. 
Your goal is to take noisy user input and extract two distinct components: 
1. The Core Query (The logical question to be answered).
2. The Local Context (Any specific facts or premises provided by the user).

### STRICT INSTRUCTIONS
- CLEANING: Fix typos, grammatical errors, and informal slang into professional, precise language.
- SEPARATION: Do not answer the question. Only extract the components.
- ATOMIZATION: Format entities to be lowercase_with_underscores to aid Prolog atomization.
- NULL HANDLING: If no context is provided, return "None".
- OUTPUT FORMAT: Return a valid JSON object.

### TASK
Analyze the user's input and structure it as follows:
{
  "rationale": "Brief explanation of your extraction process",
  "question": "The precise question for the Prolog engine",
  "context": ["List of distinct facts found in the input"],
}
'''

ENCODER_FEW_SHOT_EXAMPLES = [
#   # Example 1
#   {"role": "user", "content": "Hey so Alise is a docter and Bob is a nurce, so is Alise able to prescribe meds?"},
#   {"role": "assistant", "content": '''
# {
#   "rationale": "The user identifies two entities (Alice and Bob) and assigns them specific medical roles (Doctor and Nurse). The core of the query is a request for a permission check regarding Alice's ability to prescribe medication based on her professional status. To prepare this for the knowledge graph, I must fix the typos 'Alise' (Alice), 'docter' (doctor), 'nurce' (nurse), and 'meds' (medications).",
#   "cleaned_query": "Is Alice, who is a doctor, legally authorized to prescribe medications?",
#   "extracted_context": [
#     "Alice is a doctor.",
#     "Bob is a nurse."
#   ]
# }
# '''},
  
  # Example 2
  {"role": "user", "content": "Assuming every human is mortal, what happens to socrates?"},
  {"role": "assistant", "content": '''
{
  "rationale": "The user is presenting a logical deduction problem. The phrase 'Assuming every human is mortal' is identified as a universal premise (Local Context) that must be added to the knowledge base to facilitate the reasoning process. The second part of the sentence identifies 'Socrates' as the subject of interest and poses a question regarding his eventual state, which constitutes the core query.",
  "cleaned_query": "What is the logical consequence or fate of Socrates based on the provided premises?",
  "extracted_context": [
    "Every human is mortal.",
    "Socrates is a member of the human species."
  ]
}
'''},
  
#   # Example 3
#   {"role": "user", "content": "So, like, if most birds can fly but penguins are birds that can't, what happens to Pingu the penguin?"},
#   {"role": "assistant", "content": '''
# {
#   "rationale": "The user is asking about a specific entity 'Pingu' based on a general rule about birds and a specific exception for penguins.",
#   "cleaned_query": "What is the flight capability status of Pingu the penguin?",
#   "extracted_context": [
#     "Most birds have the ability to fly.",
#     "Penguins are a type of bird that cannot fly.",
#     "Pingu is a penguin."
#   ]
# }
# '''}
  
]