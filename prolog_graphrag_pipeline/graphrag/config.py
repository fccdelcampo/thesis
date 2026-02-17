from pathlib import Path
from enum import Enum

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

# Global Knowledge Source Configuration
GLOBAL_KB = "WIKIDATA"


class RETRIEVER_OPTIONS(Enum):
  HybridCypherRetriever = "HybridCypherRetriever"
  HybridRetriever = "HybridRetriever"
  VectorCypherRetriever = "VectorCypherRetriever"
  

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

RETRIEVER = RETRIEVER_OPTIONS.VectorCypherRetriever

# Retrieval query for HybridCypherRetriever:
# Takes the vector-matched local node and finds its local connections.


RETRIEVAL_QUERY = '''
// Part A: Local Document Retrieval (Vector/Keyword matches passed as `node`)
WITH node, score
OPTIONAL MATCH (node)-[:EXTRACTS]->(e)
OPTIONAL MATCH (e)-[r]->(target)
WITH node, score, collect(distinct {
    source: "Local Document",
    entity: e.name,
    relationship: type(r),
    target: target.name,
    text: node.text
}) as local_context

RETURN 
    node.text as text,
    score,
    {
        local_context: local_context,
        score: score
    } as metadata
'''

# Additional instructions if ever needed.
# 1. Read the Context carefully.
# 2. Identify which specific sentence supports your answer.
# 3. If the context contains multiple objects or events, do NOT mix them up. 
# 4. If the answer is not in the context, say "I don't know."

# GRAPHRAG_TEMPLATE = '''
# # INSTRUCTIONS:
# Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned.

# # Question:
# {query_text}

# # Context:
# {context}

# # Answer:
# '''

GRAPHRAG_TEMPLATE = '''
### ROLE
You are an expert Educational AI Assistant. Answer the user's question accurately using the provided Context.

### STRICT INSTRUCTIONS
1. **Answer Directly**: Start with the answer. Do NOT begin with definitions or explanations of how you found it.
2. **Sound Natural**: Write as if you already know the answer. NEVER say "the context states", "according to the provided information", "based on the evidence", or similar phrases.
3. **Use Context Silently**: The Context helps you be accurate, but the user should not know you relied on it. Integrate facts naturally.
4. **MMLU/Multiple Choice**: If options are provided (A, B, C, D), select the single best one and explain why.
5. **Never Cite Sources**: Do NOT mention Wikidata, YAGO, documents, or any source by name.

### Question:
{query_text}

### Context (for your reference only — do NOT mention it):
{context}

### Answer:
'''

DOC_PATH = Path(__file__).resolve().parents[2] / "documents"

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