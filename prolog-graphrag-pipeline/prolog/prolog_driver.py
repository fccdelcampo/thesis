import janus_swi as janus
import re
from prolog_generator import generate_prolog_code, capture_db_and_query, capture_predicate_and_arguments
from explainer import generate_safe_scasp_wrapper, generate_explanation
import time

def use_scasp():
    list(janus.query("use_module(library(scasp))."))
    janus.query_once("use_module(library(scasp/human)).")

def run_prolog_pipeline(user_question, retrieved_context) -> str:
    start_time = time.perf_counter()
    use_scasp()
    print(f"**User Question:**\n{user_question}")
    print(f"**Retrieved Context:**\n{retrieved_context}")
    print("Generating Prolog...")
    database, query = generate_prolog_code(user_question=user_question, retrieved_context=retrieved_context, most_recent_error=None)
    print(f"**Extracted database:**\n{database}")
    print(f"**Extracted query:**\n{query}")

    
    print("Generating s(CASP) wrapper...")
    wrapper = generate_safe_scasp_wrapper(query)
    print(f"**Wrapper:** \n{wrapper}")
    # final_query = f"explain({', '.join(arguments)}, Explanation)."
    final_query = f"explain(Explanation)."
    # print(f"Final query: \n{final_query}")
    
    print("Consulting database...")
    janus.consult("user", database + "\n" + wrapper)
    try:
        print("Querying database...")
        result = janus.query_once(final_query)
        print(f"**Query Results:** \n{result}")
        retrieved_values = str()
        if result:
            for arg in result.keys():  
                if arg.isupper() or arg == "truth":  # Check if it's a variable 
                    print(f"{arg} Found: {result[arg]}")    
                    retrieved_values += f"{arg}: {result[arg]} "
            # result['Tree'] will be a nested list/dictionary structure
            print("**Human-Readable Explanation**")
            human_readable_explanation = result['Explanation']
            print(human_readable_explanation)
        else:
            print("No solution found.")
    except Exception as e:
        print(f"Error: {e}")
        
    explainer_output = generate_explanation(user_question=user_question, retrieved_context=retrieved_context, database=database, query=query, retrieved_values=retrieved_values, human_readable_explanation=human_readable_explanation)
    print(f"**Explainer Output:** \n{explainer_output}")
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"====The Prolog pipeline executed in {duration:.4f} seconds.===")

    import main_driver
    print("\n**FINAL LLM OUTPUT:")
    print(main_driver.generate(user_question, retrieved_context, explainer_output))

        
        
def run_prolog_pipeline_fake(user_question, retrieved_context, database, query) -> tuple[str, str]:
    print(f"Extracted database: \n{database}")
    print(f"Extracted query: \n{query}")

    print("Generating s(CASP) wrapper...")
    wrapper = generate_safe_scasp_wrapper(query)
    print(f"Wrapper: \n{wrapper}")
    # final_query = f"explain({', '.join(arguments)}, Explanation)."
    final_query = f"explain(Explanation)."
    print(f"Final query: \n{final_query}")
    
    print("Consulting database...")
    janus.consult("user", database + "\n" + wrapper)
    try:
        print("Querying database...")
        result = janus.query_once(final_query)
        print(f"Result: \n{result}")
        retrieved_values = str()
        if result:
            for arg in result.keys():  
                if arg.isupper() or arg == "truth":  # Check if it's a variable 
                    print(f"{arg} Found: {result[arg]}")    
                    retrieved_values += f"{arg}: {result[arg]} "
            # result['Tree'] will be a nested list/dictionary structure
            print("\n--- Human-Readable Explanation ---")
            human_readable_explanation = result['Explanation']
            print(human_readable_explanation)
        
        else:
            print("No solution found.")
    except Exception as e:
        print(f"Error: {e}")
        
    return retrieved_values, human_readable_explanation, generate_explanation(user_question=user_question, retrieved_context=retrieved_context, database=database, query=query, retrieved_values=retrieved_values, human_readable_explanation=human_readable_explanation)

tests = [
    {
        "question": "Who is Alice's brother?",
        "context": "Bob is Alice's brother. Amanda is Alice's sister.",
        "database": """
male(bob).
female(alice).
female(amanda).
parent(p1, bob).
parent(p1, alice).
parent(p1, amanda).
sibling(Sibling1, Sibling2) :- parent(Parent, Sibling1), parent(Parent, Sibling2), Sibling1 \= Sibling2.
brother(Brother, Sibling) :- male(Brother), sibling(Brother, Sibling).
        """,
        "query": "brother(Brother, amanda)."
    },
    
    # Example 2
    {
        "question": "Is Maria the grandmother of Dan?",
        "context": "Maria is the mother of Chloe. Chloe is the mother of Dan.",
        "database": """
parent(maria, chloe). 
parent(chloe, dan). 
grandmother(Grandmother, Grandchild) :- parent(Grandmother, IntermediateParent), parent(IntermediateParent, Grandchild). 
        """,
        "query": "grandmother(maria, dan). "
    },
    
    # Example 3
    {
        "question": "Who does Vito believe he should keep closer?",
        "context": "Tomasinno is the friend of Vito. Solozzo is the enemy of Vito. Vito believes that he should keep his enemies closer than his friends.",
        "database": """
friend(tomasinno, vito).
enemy(solozzo, vito).
closer_than(Enemy, Friend) :- enemy(Enemy, vito), friend(Friend, vito).
vito_keeps_closer(Person) :- closer_than(Person, _). 
        """,
        "query": "vito_keeps_closer(Who)."
    },
    
    # Example 4:
    {
        "question": "Is every square a rectangle?",
        "context": "A square is a polygon with four equal sides and four right angles. A rectangle has four right angles.",
        "database": """
has_property(square, four_equal_sides). 
has_property(square, four_right_angles). 
is_a(square, polygon). 
is_a(Shape, rectangle) :- has_property(Shape, four_right_angles). 
        """,
        "query": "is_a(square, rectangle)."
    },
    
    # Example 5
    {
        "question": "If the grass is contaminated, are the snakes affected?",
        "context": "In a food web, grass is a producer. Grasshoppers eat grass. Frogs eat grasshoppers. Snakes eat frogs.",
        "database": """
eats(grasshopper, grass). 
eats(frog, grasshopper). 
eats(snake, frog). 
is_producer(grass). 
is_contaminated(grass). 
affected(Species) :- is_contaminated(Species). 
affected(Predator) :- eats(Predator, Prey), affected(Prey). 
        """,
        "query": "affected(snake)."
    },
]

instruction = """
---
### INSTRUCTION:
Generate a coherent narrative of the proof above. Start with the conclusion, then detail the step-by-step logical path taken by the solver.
"""

use_scasp()
# for test in tests:
#     retrieved_values, human_readable_explanation, decoder_explanation = run_prolog_pipeline_fake(test["question"], test["context"], test["database"], test["query"])
#     print("==========TEST LOGS==========\n")
#     print("### DECODER TASK: ELUCIDATE LOGICAL PROOF")
#     print(f"**USER QUESTION:**\n{test["question"]}\n")
#     print(f"**RETRIEVED CONTEXT:**\n{test["context"]}\n")
#     print(f"**DATABASE:**\n{test["database"]}\n")
#     print(f"**QUERY**:\n{test["query"]}\n")
#     print(f"**RETRIEVED VALUES**:\n{retrieved_values}\n")
#     print(f"**s(CASP) HUMAN-READABLE EXPLANATION:**\n{human_readable_explanation}\n")
#     print(instruction)
#     print(f"**DECODER EXPLANATION:**\n{decoder_explanation}\n")
#     print("=============================\n")
    
# Calculate the path to the directory you want to import from.
# '..' refers to the parent directory.
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(parent_dir)

import main_driver
for test in tests:
    decoder_explanation = run_prolog_pipeline(test["question"], test["context"])
    
# user_question = "Will prices go up if people are reluctant to take out loans?"
# retrieved_context = "When interest rates are high, people will be less likely to take out loans. When people take less loan out, demand for goods decrease. When demand for goods decrease, interest rates spike up. Prices go down when demand is down."
# run_prolog_pipeline(user_question, retrieved_context)

# user_question = "Who does Vito believe he should keep closer?"
# retrieved_context = "Tomasinno is the friend of Vito. Solozzo is the enemy of Vito. Vito believes that he should keep his enemies closer than his friends."
# run_prolog_pipeline(user_question, retrieved_context)

# user_question = "If the grass is contaminated, are the snakes affected?"
# retrieved_context = "In a food web, grass is a producer. Grasshoppers eat grass. Frogs eat grasshoppers. Snakes eat frogs. "
# run_prolog_pipeline(user_question, retrieved_context)

        
        
    

        
    
    
    