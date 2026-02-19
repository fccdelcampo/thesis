import janus_swi as janus
import re
import time
try:
    from .prolog_generator import generate_prolog_code, capture_db_and_query, capture_predicate_and_arguments
    from .explainer import generate_safe_scasp_wrapper, generate_explanation
    from .. import llm
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from prolog_graphrag_pipeline.prolog.prolog_generator import generate_prolog_code, capture_db_and_query, capture_predicate_and_arguments
    from prolog_graphrag_pipeline.prolog.explainer import generate_safe_scasp_wrapper, generate_explanation
    import prolog_graphrag_pipeline.llm as llm

SCASP_AVAILABLE = True

def use_scasp():
    global SCASP_AVAILABLE
    try:
        list(janus.query("use_module(library(scasp))."))
        janus.query_once("use_module(library(scasp/human)).")
        print("s(CASP) loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: s(CASP) library not found: {e}")
        SCASP_AVAILABLE = False

def run_pipeline(user_question: str, retrieved_context: str, context_from_prompt: str) -> str:
    start_time = time.perf_counter()
    # Ensure inputs are strings
    if isinstance(retrieved_context, list):
        retrieved_context = "\n".join(retrieved_context)
    if not retrieved_context:
        retrieved_context = ""
        
    if isinstance(context_from_prompt, list):
        context_from_prompt = "\n".join(context_from_prompt)
    if not context_from_prompt:
        context_from_prompt = ""

    print(f"**User Question:**\n{user_question}")
    print(f"**Retrieved Context:**\n{retrieved_context}")
    print(f"**Context from Prompt:**\n{context_from_prompt}")
    print("Generating Prolog...")
    final_context = retrieved_context+ "\n" +context_from_prompt
    database, query = generate_prolog_code(user_question=user_question, retrieved_context=final_context, most_recent_error=None)
    print(f"**Extracted database:**\n{database}")
    print(f"**Extracted query:**\n{query}")

    if not SCASP_AVAILABLE:
        raise RuntimeError("s(CASP) library is specifically required for this pipeline. Execution aborted.")

    wrapper = ""
    human_readable_explanation = ""
    retrieved_values = ""

    print("Generating s(CASP) wrapper...")
    wrapper = generate_safe_scasp_wrapper(query)
    print(f"**Wrapper:** \n{wrapper}")
    final_query = f"explain(Explanation)."
    
    print("Consulting database...", flush=True)
    janus.consult("user", database + "\n" + wrapper)
    try:
        print("Querying database...", flush=True)
        result = janus.query_once(final_query)
        print(f"**Query Results:** \n{result}", flush=True)
        
        if result:
            for arg in result.keys():  
                if arg not in ["Explanation", "Tree", "Model"] and (arg.isupper() or arg == "truth"):  
                    print(f"{arg} Found: {result[arg]}")    
                    retrieved_values += f"{arg}: {result[arg]} "
            
            if SCASP_AVAILABLE and 'Explanation' in result:
                print("**Human-Readable Explanation**")
                human_readable_explanation = result['Explanation']
                print(human_readable_explanation)
        else:
            print("No solution found.")
    except Exception as e:
        print(f"Error: {e}")
        
    explainer_output = generate_explanation(user_question=user_question, retrieved_context=retrieved_context, database=database, query=query, retrieved_values=retrieved_values, human_readable_explanation=human_readable_explanation)
    if explainer_output:
        print(f"**Explainer Output:** \n{explainer_output}")
    else:
        print("ERROR: Explainer failed to generate output.")

    print("\n**FINAL LLM OUTPUT:")
    final_answer = llm.generate(user_question, retrieved_context, explainer_output, flag="synthesis")
    
    if final_answer:
        print(final_answer["text_answer"])
    else:
        print("ERROR: Final LLM failed to generate answer.")
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"====The Prolog pipeline executed in {duration:.4f} seconds.===")
    return final_answer if final_answer else {"text_answer": "Error: Final LLM failed to generate answer."}


if __name__ == "__main__":
    tests = [
        {
            "question": "Who is Alice's brother?",
            "context": "Bob is Alice's brother. Amanda is Alice's sister.",
            "database": rf"""
    male(bob).
    female(alice).
    female(amanda).
    parent(p1, bob).
    parent(p1, alice).
    parent(p1, amanda).
    sibling(Sibling1, Sibling2) :- parent(Parent, Sibling1), parent(Parent, Sibling2), Sibling1 \= Sibling2.
    brother(Brother, Sibling) :- male(Brother), sibling(Brother, Sibling).
            """,
            "query": rf"brother(Brother, amanda)."
        },
        
        # Example 2
        {
            "question": "Is Maria the grandmother of Dan?",
            "context": "Maria is the mother of Chloe. Chloe is the mother of Dan.",
            "database": rf"""
    parent(maria, chloe). 
    parent(chloe, dan). 
    grandmother(Grandmother, Grandchild) :- parent(Grandmother, IntermediateParent), parent(IntermediateParent, Grandchild). 
    """,
            "query": rf"grandmother(maria, dan). "
        },
        
        # Example 3
        {
            "question": "Who does Vito believe he should keep closer?",
            "context": "Tomasinno is the friend of Vito. Solozzo is the enemy of Vito. Vito believes that he should keep his enemies closer than his friends.",
            "database": rf"""
    friend(tomasinno, vito).
    enemy(solozzo, vito).
    closer_than(Enemy, Friend) :- enemy(Enemy, vito), friend(Friend, vito).
    vito_keeps_closer(Person) :- closer_than(Person, _). 
    """,
            "query": rf"vito_keeps_closer(Who)."
        },
        
        # Example 4:
        {
            "question": "Is every square a rectangle?",
            "context": "A square is a polygon with four equal sides and four right angles. A rectangle has four right angles.",
            "database": rf"""
    has_property(square, four_equal_sides). 
    has_property(square, four_right_angles). 
    is_a(square, polygon). 
    is_a(Shape, rectangle) :- has_property(Shape, four_right_angles). 
    """,
            "query": rf"is_a(square, rectangle)."
        },
        
        # Example 5
        {
            "question": "If the grass is contaminated, are the snakes affected?",
            "context": "In a food web, grass is a producer. Grasshoppers eat grass. Frogs eat grasshoppers. Snakes eat frogs.",
            "database": rf"""
    eats(grasshopper, grass). 
    eats(frog, grasshopper). 
    eats(snake, frog). 
    is_producer(grass). 
    is_contaminated(grass). 
    affected(Species) :- is_contaminated(Species). 
    affected(Predator) :- eats(Predator, Prey), affected(Prey). 
    """,
            "query": rf"affected(snake)."
        },
    ]

    use_scasp()
    # Calculate the path to the directory you want to import from.
    # '..' refers to the parent directory.

    for test in tests[:1]:
        decoder_explanation = run_pipeline(test["question"], test["context"], "")
    
# user_question = "Will prices go up if people are reluctant to take out loans?"
# retrieved_context = "When interest rates are high, people will be less likely to take out loans. When people take less loan out, demand for goods decrease. When demand for goods decrease, interest rates spike up. Prices go down when demand is down."
# run_prolog_pipeline(user_question, retrieved_context)

# user_question = "Who does Vito believe he should keep closer?"
# retrieved_context = "Tomasinno is the friend of Vito. Solozzo is the enemy of Vito. Vito believes that he should keep his enemies closer than his friends."
# run_prolog_pipeline(user_question, retrieved_context)

# user_question = "If the grass is contaminated, are the snakes affected?"
# retrieved_context = "In a food web, grass is a producer. Grasshoppers eat grass. Frogs eat grasshoppers. Snakes eat frogs. "
# run_prolog_pipeline(user_question, retrieved_context)

        
        
    

        
    
    
    