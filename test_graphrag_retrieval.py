import json

from prolog_graphrag_pipeline import main_driver

def retrieve_context(formatted_input: str, debug_id: int) -> None:
    # In a real implementation, this would query a vector database or search engine
    print(f"=============================\nRetrieving context for: {formatted_input}")
    retrieved_context = main_driver.run_graphrag_pipeline(user_prompt=formatted_input)
    print(f"Done retrieving context.")
    log_retrieved_context(formatted_input, retrieved_context, debug_id)
    print(f"Debug test case #{debug_id} done.\n=============================\n")
    return retrieved_context

def log_retrieved_context(test_case: dict, retrieved_context: dict, debug_id: int) -> None:
    """Log the retrieved context for debugging purposes."""
    formatted_input = test_case["formatted_input"]
    correct_answer = test_case["correct_answer"]
    benchmark_name = test_case["dataset"]
    with open("graphrag_retrieval.log", "a", encoding="utf-8") as log_file:
        log_file.write(f"   [+] Debug ID #{debug_id} | Benchmark: {benchmark_name}\n      Formatted Input: {formatted_input}\n        Context: {retrieved_context}\n        Correct Answer: {correct_answer}\n")
        log_file.flush()
    print(f"   [+] Debug ID #{debug_id} | Benchmark: {benchmark_name}\n      Formatted Input: {formatted_input}\n        Context: {retrieved_context}\n        Correct Answer: {correct_answer}\n")
    print(f"Retrieved context logged for debug ID {debug_id}.")
    
def run_tests():
    test_cases_ai2_arc, test_cases_openbookqa = [], []
    with open("test_cases/ai2_arc.jsonl", "r", encoding="utf-8") as f:
        for line in f[:20]:
            test_cases_ai2_arc.append(json.loads(line))
            
    with open("test_cases/openbookqa.jsonl", "r", encoding="utf-8") as f:
        for line in f[:20]:
            test_cases_openbookqa.append(json.loads(line))
            
    for i, test_case in enumerate(test_cases_ai2_arc + test_cases_openbookqa):
        retrieve_context(test_case["formatted_input"], debug_id=i+1)