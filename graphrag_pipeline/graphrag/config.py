from pathlib import Path
from enum import Enum

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

class RETRIEVER_OPTIONS(Enum):
  HybridCypherRetriever = "HybridCypherRetriever"
  HybridRetriever = "HybridRetriever"
  VectorCypherRetriever = "VectorCypherRetriever"
  
RETRIEVER = RETRIEVER_OPTIONS.VectorCypherRetriever

ENABLE_WIKIDATA_FETCHING = True
ENABLE_LLM_FILTERING = True

class SCHEMA_TYPE(Enum):
  STATIC = "Static Schema"
  EXTRACTED = "Extracted Schema"
  
SCHEMA_CONFIG = SCHEMA_TYPE.STATIC

STATIC_SCHEMA = {
    "node_types": [
        "ReferenceConcept",  
        "Person",            
        "Event",             
        "Denotative",        
        "Associative",       
        "ConceptualSystem"   
    ],
    "relationship_types": [
        "IS_CLOSELY_RELATED", # KKO equivalent for 'RELATED_TO'
        "SUBCLASS_OF",        # KKO strict Type-to-Type hierarchy
        "INSTANCE_OF",        # KKO strict Entity-to-Type instantiation
        "PART_OF",            # KKO mereological parthood
        "CAUSES",             
        "HAS_DEFINITION",     # KKO bridge to Denotatives
        "HAS_EXAMPLE"         # KKO bridge to Associatives
    ],
    "patterns": [
        # Defining these valid routing paths is critical so the extracted 
        # graph translates cleanly into deterministic logical facts.
        ("ReferenceConcept", "SUBCLASS_OF", "ConceptualSystem"),
        ("ConceptualSystem", "SUBCLASS_OF", "ConceptualSystem"),
        ("Person", "INSTANCE_OF", "ReferenceConcept"),
        ("Event", "INSTANCE_OF", "ReferenceConcept"),
        ("Event", "CAUSES", "Event"),
        ("ReferenceConcept", "PART_OF", "ReferenceConcept"),
        ("ReferenceConcept", "IS_CLOSELY_RELATED", "ReferenceConcept"),
        ("ReferenceConcept", "HAS_DEFINITION", "Denotative"),
        ("ReferenceConcept", "HAS_EXAMPLE", "Associative")
    ],
    "additional_node_types": False,
    "additional_relationship_types": False
}

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

# Retrieval query for HybridCypherRetriever/VectorCypherRetriever:
# Takes the vector-matched local node and returns it as a standard chunk.
RETRIEVAL_QUERY = '''
// Part A: Local Document Retrieval (Vector matches passed as `node`)
RETURN 
    node.text as text,
    score,
    {
        score: score,
        source: "Local Document"
    } as metadata
'''

# Additional instructions if ever needed.
# 1. Read the Context carefully.
# 2. Identify which specific sentence supports your answer.
# 3. If the context contains multiple objects or events, do NOT mix them up. 
# 4. If the answer is not in the context, say "I don't know."

GRAPHRAG_TEMPLATE = '''
### ROLE
You are an expert Educational AI Assistant answering a student's question based ONLY on the provided Context.

### STRICT INSTRUCTIONS
1. **Be Direct**: Output your final answer immediately, concisely, and clearly. Do NOT start with "Based on the context" or "The answer is...". 
2. **If Multiple Choice**: Select the correct option letter and give a brief 1-2 sentence justification combining the facts from the Context.
3. **Ignore Irrelevant Facts**: The context might contain extra noisy concepts. Only use the facts that logically connect to solve the query.
4. **No Hallucinations**: If the context completely fails to provide a logical bridge to the answer, strictly write: "I do not have enough information to answer this question."
5. **Format Consistency**: Do NOT provide conversational filler (e.g., "The final answer is not provided..."). Answer the prompt directly.

### Question:
{query_text}

### Context:
{context}

### Answer:
'''

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
ENCODER_SYSTEM_PROMPT = '''
### ROLE
You are a Query Pre-processor for a Retrieval System. 
Your goal is to split the user's input into:
1. The Core Question (to be sent to the search engine).
2. The Context (any background facts/premises provided in the prompt).

### STRICT INSTRUCTIONS
- IF MULTIPLE CHOICE (A, B, C, D): You MUST preserve the options VERBATIM in the 'question' field. Do NOT strip them.
- CLEANING: Fix typos but keep the original meaning.
- SEPARATION: 
  - 'question': The actual question asked.
  - 'context': A list of sentences/facts provided as background.
- NULL HANDLING: If no context is provided, return an empty list [].
- OUTPUT FORMAT: Return a valid JSON object.

### TASK
Analyze the user's input and structure it as follows:
{
  "rationale": "Brief explanation",
  "question": "The precise question (including options if present)",
  "context": ["Fact 1", "Fact 2"]
}
'''

ENCODER_FEW_SHOT_EXAMPLES = [
  # Example 1: Standard QA
  {"role": "user", "content": "Alise is a doctor and Bob is a nurse. Can Alise prescribe meds?"},
  {"role": "assistant", "content": '''
{
  "rationale": "Separating background facts about professions from the specific question about prescribing rights.",
  "question": "Can Alise prescribe medications?",
  "context": [
    "Alise is a doctor.",
    "Bob is a nurse."
  ]
}
'''},
  
  # Example 2: Logic Puzzle
  {"role": "user", "content": "Assuming every human is mortal, what happens to Socrates?"},
  {"role": "assistant", "content": '''
{
  "rationale": "Extracting the premise about mortality as context and the specific inquiry about Socrates as the question.",
  "question": "What happens to Socrates?",
  "context": [
    "Every human is mortal."
  ]
}
'''},
  
  # Example 3: Multiple Choice (MMLU Style)
  {"role": "user", "content": "What is the capital of France? A) London B) Paris C) Berlin D) Madrid"},
  {"role": "assistant", "content": '''
{
  "rationale": "This is a multiple choice question. The options must be preserved in the question field.",
  "question": "What is the capital of France? A) London B) Paris C) Berlin D) Madrid",
  "context": []
}
'''}
]
