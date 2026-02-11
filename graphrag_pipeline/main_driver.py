from graphrag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from graphrag.graphrag_driver import *
from graphrag.retriever import create_retriever
from graphrag.encoder import extract_query_and_context, process_context
from llm import generate
import neo4j
import json

neo4j_driver = neo4j.GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

encoder_llm, retriever_llm, embedder = initialize_models()

kg_builder_pdf, kg_builder_text = setup_kg_pipeline(encoder_llm, embedder)

retriever = create_retriever(neo4j_driver, embedder)

def run_pipeline(user_question):
    return generate(retriever_llm, retriever, user_question)

run_pipeline("What physical obstacle prevented Alice from retrieving the item she found on the three-legged table, immediately after she obeyed the command printed on the bottle label?")
    
# def load_test_cases(name: str, start: int, end: int) -> dict:
#     test_cases = []
#     try:
#         with open(f'test_cases/{name}.jsonl', 'r', encoding="utf-8") as file:
#             for line in file[start: end+1]:
#                 test_cases.append(json.load(line))
#         return test_cases
#     except FileNotFoundError:
#         print(f"Error: The file '{name}.json' was not found.")
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from the file '{name}.json'. Check file format.")
#     except Exception as e:
#         print(f"Error loading test cases: {e}")
        
# def run_test_case(test_case: dict):
#     query, context = extract_query_and_context(test_case.get("query", ""))
    
#     asyncio.run(process_context(kg_builder_pdf, kg_builder_text, [context]))
    
#     return generate_answer(query, retriever, retriever_llm)

# def run_test_cases(test_cases: dict, dataset_name: str):
#     dialogues = []
#     for test_case in test_cases:
#         response = run_test_case(test_case)
#         dialogues.append(
#             {
#                 "question": test_case["question"],
#                 "answer": response["answer"],
#                 "context": response["context"]
#             }
#         )
        
#     log_dialogues(dialogues)
    
# def log_dialogues(dialogues: list[dict], dataset_name):
#     with open(f"/logs/{dataset_name}.jsonl", "a", encoding="utf-8") as file:
#         for dialogue in dialogues:
#             file.write(json.dumps(dialogue, ensure_ascii=False) + "\n")