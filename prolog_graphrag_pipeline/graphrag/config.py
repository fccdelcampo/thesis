from pathlib import Path
from enum import Enum

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

# Global Knowledge Source Configuration
GLOBAL_KB = "KBPEDIA"

# Wikidata Fallback: When KBPedia has no match for an entity,
# fall back to Wikidata REST Search API + structural SPARQL.
ENABLE_WIKIDATA_FALLBACK = True

# Coarse concept-level filtering before triple filtering.
# Default: False (faster, collapses redundant LLM turn).
ENABLE_KBPEDIA_CONCEPT_FILTER = True
ENABLE_LLM_FILTERING = True


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
You are a Knowledge Engineer. Extract entities and relationships from the text to build a Prolog-compatible Knowledge Graph.

For each relationship include: logic_intent (FACT | RULE | EXCEPTION) and a natural_statement preserving the original phrasing.

### JSON FORMAT:
{{
  "nodes": [{{"id": "atom_id", "label": "PrologLabel", "properties": {{"name": "Name"}}}}],
  "relationships": [{{
    "type": "PREDICATE_TYPE",
    "start_node_id": "id1",
    "end_node_id": "id2",
    "properties": {{
      "natural_statement": "full extracted fact or rule",
      "logic_intent": "FACT | RULE | EXCEPTION",
      "logical_form_hint": "e.g., mortal(X) :- man(X)"
    }}
  }}]
}}

### Constraints
- IDs and types: lowercase_with_underscores (Prolog atom style).
- Do not hallucinate. Return empty arrays if no clear logic exists.
- {schema}

### Examples
{examples}

### Input Text
{text}

Output JSON ONLY. No markdown, no preamble.
'''

RETRIEVER = RETRIEVER_OPTIONS.VectorCypherRetriever

RETRIEVAL_QUERY = '''
// Part A: Local Document Retrieval (Vector/Keyword matches passed as `node`)
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
You are a Logic Discovery Agent. Extract the factual and structural logical building blocks needed to answer the Question from the Context.

### TASK
Produce a single numbered list called **LOGICAL EVIDENCE**. 
First, gather evidence and definitions for all concepts related to the Question, especially focusing on each of the provided MCQ choices. 
Then, add other structural logical evidence that bridges the premise to the answer.
Each item must be a factual sentence — either a direct fact or a conditional rule (If-Then). Do NOT separate facts and rules into different sections. Do NOT repeat or paraphrase the same idea.

### CONSTRAINTS
* Include at most 15 items total. Keep them highly concise but strictly factual.
* Each item must directly help bridge the premise to the answer. Omit background, history, or tangential details.
* If the context is incomplete, you may add ONE universally accepted truth as the "missing link."
* Break compound sentences into independent logical units.
* Use clear, declarative English only. No code, no symbols.
* Do NOT reveal the final answer — provide logical ingredients only.



---
### QUESTION:
{query_text}

### CONTEXT:
{context}

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