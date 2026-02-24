import json

from prolog_graphrag_pipeline import main_driver

def retrieve_context(test_case: dict, debug_id: int) -> None:
    # In a real implementation, this would query a vector database or search engine
    formatted_input = test_case["formatted_input"]
    print(f"=============================\nRetrieving context for: {formatted_input}")
    retrieved_context = main_driver.run_graphrag_pipeline(user_prompt=formatted_input)
    print(f"Done retrieving context.")
    log_retrieved_context(test_case, retrieved_context, debug_id)
    print(f"Debug test case #{debug_id} done.\n=============================\n")
    return retrieved_context

def log_retrieved_context(test_case: dict, retrieved_context: dict, debug_id: int) -> None:
    """Log the retrieved context for debugging purposes."""
    formatted_input = test_case["formatted_input"]
    correct_answer = test_case["correct_answer"]
    benchmark_name = test_case["dataset"]
    
    if isinstance(retrieved_context, dict):
        # llm_answer          = retrieved_context.get("answer", "N/A")
        context_strings     = retrieved_context.get("contexts", [])
        # context_from_prompt = retrieved_context.get("context_from_prompt", [])
        prolog_error        = retrieved_context.get("prolog_error", None)  # None = verified
    else:
        # llm_answer          = "N/A"
        context_strings     = retrieved_context
        # context_from_prompt = []
        prolog_error        = None

    # Append a non-disruptive verification note when Prolog failed
    verification_note = ""
    if prolog_error is not None:
        verification_note = f"\n        [⚠ Prolog verification FAILED — unverified answer. Reason: {prolog_error}]"

    with open("kbpedia_graphrag_retrieval.log", "a", encoding="utf-8") as log_file:
        log_file.write(
            f"   [+] Debug ID #{debug_id} | Benchmark: {benchmark_name}\n"
            f"      Formatted Input: {formatted_input}\n"
            # f"        Answer:            {llm_answer}{verification_note}\n"
            f"        Context (retrieved): {context_strings}\n"
            # f"        Context (ingested):  {context_from_prompt}\n"
            f"        Correct Answer: {correct_answer}\n"
        )
        log_file.flush()
    print(
        f"   [+] Debug ID #{debug_id} | Benchmark: {benchmark_name}\n"
        f"      Formatted Input: {formatted_input}\n"
        # f"        Answer:            {llm_answer}{verification_note}\n"
        f"        Context (retrieved): {context_strings}\n"
        # f"        Context (ingested):  {context_from_prompt}\n"
        f"        Correct Answer: {correct_answer}\n"
    )
    print(f"Retrieved context logged for debug ID {debug_id}.")

    
def run_tests(start_from_id=1):
    test_cases_ai2_arc, test_cases_openbookqa = [], []
    with open("test_cases/ai2_arc.jsonl", "r", encoding="utf-8") as f:
        for line in f.readlines()[:20]:
            test_cases_ai2_arc.append(json.loads(line))
            
    with open("test_cases/openbookqa.jsonl", "r", encoding="utf-8") as f:
        for line in f.readlines()[:20]:
            test_cases_openbookqa.append(json.loads(line))
            
    all_tests = test_cases_ai2_arc + test_cases_openbookqa
    
    for i, test_case in enumerate(all_tests):
        debug_id = i + 1
        if debug_id < start_from_id:
            continue
        try:
            retrieve_context(test_case, debug_id=debug_id)
        except Exception as e:
            print(f"[ERROR] Debug ID #{debug_id} failed: {e}")
            with open("kbpedia_graphrag_retrieval.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"   [ERROR] Debug ID #{debug_id} | SKIPPED due to: {e}\n")
            continue


run_tests(start_from_id=21)