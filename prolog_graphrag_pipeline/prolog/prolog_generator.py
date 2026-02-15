import janus_swi as janus
import re

def capture_db_and_query(answer_text: str) -> tuple[str, str]:
    import re

    db_match = re.search(r"<database>(.*?)</database>", answer_text, re.DOTALL)
    # if "</query>" not in answer_text:
    #     answer_text += "</query>"
    query_match = re.search(r"<query>(.*?)</query>", answer_text, re.DOTALL)

    try:
        db = db_match.group(1)
    except:
        raise ValueError(f"Database not found in LLM's response \n{answer_text}")
    try:
        query = query_match.group(1)
    except:
        raise ValueError(f"Query not found in LLM's response: \n{answer_text}")

    return {"database":db, "query":query}


def capture_predicate_and_arguments(query: str) -> dict:
    
    predicate_match = re.search(r"(\w+)\s*\(", query)
    try:
        predicate = predicate_match.group(1)
    except:
        raise ValueError("Could not extract predicate name from query")
    
    query = query.replace(".", "").replace("\n", "")
    while query.count("(") > 0 and query.count("(") == query.count(")"):
        ei = query.find("(")
        print(f"prefix: {query[:ei+1]}")
        query = query.replace(query[:ei+1], "").removesuffix(")")
    
    arguments_match = re.search(r"\((.*?)\)", query)
    try:
        arguments = arguments_match.group(1).replace(" ", "").split(",")
        print("Extracted Arguments:", arguments)
    except:
        raise ValueError("Could not extract arguments from query")
    return {"predicate": predicate, "arguments": arguments}

def generate_prolog_generation_prompt(user_question: str, retrieved_context: str, most_recent_error: str) -> str:
    prompt = f"""
Now, process the following:
Context:
{retrieved_context}

User Question: 
{user_question}
"""

    if most_recent_error:
        prompt += f"""
Your recent Prolog database and query have errors. Fix them based on the following error message:
{most_recent_error}
"""

    return prompt

from typing import Optional

def generate_prolog_code(user_question: str, retrieved_context: str, most_recent_error: Optional[str]) -> tuple:
    for i in range(5):
        try:
            try:
                from .prolog_llms import generate
            except ImportError:
                import sys
                import os
                # Add the project root to sys.path
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                from prolog_graphrag_pipeline.prolog.prolog_llms import generate
            answer = generate(prompt=generate_prolog_generation_prompt(user_question, retrieved_context, most_recent_error), flag="prolog")
            if not answer:
                print(f"Iteration {i} | Error: LLM returned None.")
                continue
            database, query = capture_db_and_query(answer['text_answer'])["database"], capture_db_and_query(answer['text_answer'])["query"]
            print(f"Iteration {i} | Extracted Database:\n{database}\n")
            print(f"Iteration {i} | Extracted Query:\n{query}\n")
            janus.query_once("unload_file(user)")
            janus.consult("user", database)
            janus.query_once(query)
            most_recent_error = None
            print(f"Iteration {i} | Prolog code generation success")
            return database, query
        # except Exception as e:
        #     print(f"Iteration {i} | Error during prompt generation or model interaction: {e}")
        #     continue
        except janus.PrologError as e:
            print(f"Iteration {i} | Prolog Error: {e}")
            most_recent_error = str(e + " Ensure all Prolog syntax rules are followed. Enclose the final database within <database> and </database> tags, and the query within <query> and </query> tags.")
            print(most_recent_error)
            continue
        
    if most_recent_error:
        raise Exception("Failed to generate valid Prolog code after multiple attempts.")
