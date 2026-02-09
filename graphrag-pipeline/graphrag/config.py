from pathlib import Path
from enum import Enum

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

class RETRIEVER_OPTIONS(Enum):
  HybridCypherRetriever = "HybridCypherRetriever"
  HybridRetriever = "HybridRetriever"
  
RETRIEVER = RETRIEVER_OPTIONS.HybridRetriever

class SCHEMA_TYPE(Enum):
  STATIC = "Static Schema"
  EXTRACTED = "Extracted Schema"
  
SCHEMA_CONFIG = SCHEMA_TYPE.STATIC

STATIC_SCHEMA = { # NOTE: Due to how 'EXTRACTED' works, the prompt used by SimpleKGPipeline is bad and does not constrain the output to only JSON format. We can create a custom KG Pipeline but for now we use this to skip it entirely.
        "node_types": [
            "Concept",      # The heavy lifter: covers Theorems, Ideas, Theories, Terms
            "Person",       # Key figures: Socrates, Einstein, Historical Leaders
            "Event",        # Occurrences: Wars, Experiments, Historical Periods
            "Definition",   # Crucial for education: Explicit statements of meaning
            "Example",      # Grounding: Specific instances (e.g., "Pingu" or "3")
            "Topic"         # The Field: "Mathematics", "Philosophy", "Biology"
        ],
        "relationship_types": [
            "RELATED_TO",   # Generic connection
            "IS_A",         # Hierarchy: "Calculus IS_A Topic", "Penguin IS_A Bird"
            "PART_OF",      # Composition: "Cell PART_OF Tissue"
            "CAUSES",       # Logic/History: "War CAUSES Famine", "Premise CAUSES Conclusion"
            "DEFINES",      # Concept -> Definition
            "ILLUSTRATES"   # Example -> Concept ("Pingu ILLUSTRATES Non-flying bird")
        ]
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

RETRIEVAL_QUERY = '''
"""
MATCH (node)-[:EXTRACTS]->(e:Entity)
OPTIONAL MATCH (e)-[r]->(target:Entity)
RETURN node.text AS text, 
       node.score AS score,
       {entity: e.name, relationship: type(r), connected_to: target.name} AS graph_metadata
"""
'''
# Additional instructions if ever needed.
# 1. Read the Context carefully.
# 2. Identify which specific sentence supports your answer.
# 3. If the context contains multiple objects or events, do NOT mix them up. 
# 4. If the answer is not in the context, say "I don't know."
GRAPHRAG_TEMPLATE = '''
# INSTRUCTIONS:
Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned.

# Question:
{query_text}

# Context:
{context}

# Answer:
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