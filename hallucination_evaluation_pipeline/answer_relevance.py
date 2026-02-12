def load_test_cases(path: str) -> list[dict]:
    pass

def run_test_case(test_case: dict):
    # // run LLM as judge
    log_results(test_case_id, score)
    question: str = None
    pipeline.run_pipeline(question)
    pass

def run_test_cases(test_cases: list[dict]):
    for t in test_cases:
        run_test_case(t)
    pass

def log_results(test_case_id: int, score: float) -> None:
    pass

def main():
    path = r"../../test_cases/answer_relevance.jsonl"
    test_cases = load_test_cases(path)
    run_test_cases(test_cases)