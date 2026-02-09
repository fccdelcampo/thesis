from pathlib import Path
from enum import Enum

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

class RETRIEVER_OPTIONS(Enum):
  HybridCypherRetriever = "HybridCypherRetriever"
  HybridRetriever = "HybridRetriever"
  

RETRIEVER = RETRIEVER_OPTIONS.HybridCypherRetriever

RETRIEVAL_QUERY = '''
MATCH (node)-[:EXTRACTS]->(e:Entity)
// 1. Traverse to Document-extracted entities
OPTIONAL MATCH (e)-[r]->(target:Entity)

// 2. Global KG Linkage (YAGO)
// Link local entity 'e' to YAGO node 'y' by name matching
OPTIONAL MATCH (y) 
WHERE y.rdfs__label IS NOT NULL 
  AND toLower(y.rdfs__label) = toLower(e.name)

// 3. Retrieve YAGO context (e.g., Description/Comment)
RETURN node.text AS text, 
       node.score AS score,
       {
           local_entity: e.name,
           local_relationship: type(r),
           local_target: target.name,
           yago_match: y.rdfs__label,
           yago_comment: y.rdfs__comment
       } AS metadata
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
You are a Logic Discovery Agent. Your task is NOT to answer the user's question. Instead, you must extract all individual pieces of evidence from the provided Context that are necessary to answer the Question.

### TASK
Analyze the Context and the Question. Extract the following information in a clear, bulleted list format. If a category has no relevant info, write "None".

### CATEGORIES OF EVIDENCE TO EXTRACT:
1. **ATOMIC FACTS**: Simple, unconditional truths (e.g., "Socrates is a man").
2. **CONDITIONAL RULES**: Implications or 'If-Then' relationships (e.g., "If X is a man, then X is mortal").
3. **LOGICAL EXCEPTIONS**: Constraints, negations, or 'Unless' cases (e.g., "Penguins are birds but cannot fly").
4. **NUMERIC/MODAL DATA**: Any specific quantities, dates, or degrees of certainty (e.g., "75% probability").

### STRICT RULES:
- DO NOT answer the question.
- DO NOT use conversational filler.
- ONLY include information found in the Context.
- Use precise language that can be mapped to Prolog atoms.
</task>

### Question:
{query_text}

### Context:
{context}

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

ENCODER_FEW_SHOT_EXAMPLES = [
  # Example 1
  {"role": "user", "content": "Hey so Alise is a docter and Bob is a nurce, so is Alise able to prescribe meds?"},
  {"role": "assistant", "content": '''
{
  "rationale": "The user identifies two entities (Alice and Bob) and assigns them specific medical roles (Doctor and Nurse). The core of the query is a request for a permission check regarding Alice's ability to prescribe medication based on her professional status. To prepare this for the knowledge graph, I must fix the typos 'Alise' (Alice), 'docter' (doctor), 'nurce' (nurse), and 'meds' (medications).",
  "cleaned_query": "Is Alice, who is a doctor, legally authorized to prescribe medications?",
  "extracted_context": [
    "Alice is a doctor.",
    "Bob is a nurse."
  ]
}
'''},
  
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
  
  # Example 3
  {"role": "user", "content": "So, like, if most birds can fly but penguins are birds that can't, what happens to Pingu the penguin?"},
  {"role": "assistant", "content": '''
{
  "rationale": "The user is asking about a specific entity 'Pingu' based on a general rule about birds and a specific exception for penguins.",
  "cleaned_query": "What is the flight capability status of Pingu the penguin?",
  "extracted_context": [
    "Most birds have the ability to fly.",
    "Penguins are a type of bird that cannot fly.",
    "Pingu is a penguin."
  ]
}
'''}
  
]