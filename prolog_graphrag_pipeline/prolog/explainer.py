import janus_swi as janus

def generate_explanation_prompt(user_question: str, retrieved_context: str, retrieved_values: str) -> str:
    prompt = f"""
Your Prolog database has been successfully generated and consulted. Now, provide a detailed, logical, and step-by-step human-readable explanation of the justification Prolog gave. It should serve as context for answering the user's question.
User Question:
{user_question}

Context:
{retrieved_context}

The values retrieved from the query are:
{retrieved_values}
"""

    return prompt

def generate_safe_scasp_wrapper(query):
    query = query.replace(".", "").replace("\n", "")
# explain({args_str}, Explanation) :-
    return rf"""
explain(Explanation) :-
    % 1. Run the solver
    scasp({query}, [tree(Tree)]),
    
    % 2. Capture the human-readable output into a string
    % We call human_justification_tree INSIDE with_output_to
    with_output_to(string(Explanation), 
        human_justification_tree(Tree, [tree_style(human)])
    ).
"""

def generate_explanation_prompt(user_question: str, retrieved_context: str, database: str, query: str, human_readable_explanation: str, retrieved_values: str) -> str:
    prompt = f"""
### DECODER TASK: ELUCIDATE LOGICAL PROOF
**USER QUESTION:**
{user_question}

**RETRIEVED CONTEXT:**
{retrieved_context}

**DATABASE:**
{database}
        
**QUERY**:
{query}

**RETRIEVED VALUES**:
{retrieved_values} 

**s(CASP) HUMAN-READABLE EXPLANATION:**
{human_readable_explanation}

---
### INSTRUCTION:
Generate a coherent narrative of the proof above. Start with the conclusion, then detail the step-by-step logical path taken by the solver.
"""
    return prompt

def generate_explanation(user_question: str, retrieved_context: str, database: str, query: str, retrieved_values: str, human_readable_explanation: str) -> str:
    from .prolog_llms import generate
    explanation = generate(prompt=generate_explanation_prompt(user_question=user_question, retrieved_context=retrieved_context, database=database, query=query, retrieved_values=retrieved_values, human_readable_explanation=human_readable_explanation), flag="explanation")
    try:
        return explanation['text_answer']
    except Exception as e:
        print(f"Error during explanation generation: {e}")
        return None