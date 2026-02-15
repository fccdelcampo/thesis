def load_test_cases() -> list[dict]:
    """
    Reads .jsonl files in test_cases/ and turns each JSON line into a dict.

    Args:
        None. Path must be defined in the function itself.

    Returns:
        list[dict]: A list of dict objects where each dict is a test case containing, at the minimum, the question and answer fields.
    """
    return []

def run_test_case(test_case: dict):
    """
    Runs a test case by calling each pipeline's run_pipeline function. 
    Calls necessary LLM-as-judge evaluations after pipeline returns answer dict.
    Calls log_question_and_answer() to log the question and answer for each test case.
    Calls log_results() after all evaluations have returned scores

    Args:
        test_case (dict): A test case's dict
    """
    pass


from typing import Union

def log_results(test_case_id: int, tuned_score: Union[float, int], graphrag_score: Union[float, int], prolog_graphrag_score: Union[float, int]) -> None:
    """
    Logs results of the evaluations in the appropriate CSV files in hallucination_evaluation_results/
    Must automatically infer the 0/1 label and also record them in the CSV files
    Each file in hallucination_evaluation_results/ has 7 fields: 
        test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label

    Args:
        test_case_id (int): The ID of a test case
        tuned_score (float | int): The score attained by the instruction-tuned LLM pipeline given by the LLM-as-judge.
        graphrag_score (float | int): The score attained by the GraphRAG pipeline given by the LLM-as-judge.
        prolog_graphrag_score (float | int): The score attained by the Prolog-GraphRAG pipeline given by the LLM-as-judge.
    """
    pass


def log_question_and_answer(test_case_id: int, question: str, answer: str, score: Union[int, float], label: str) -> None:
    """
    Logs the question and answer for a given test case ID to a CSV file in logs/{pipeline}_logs/.
    Each log is a JSON line.

    Args:
        test_case_id (int): The ID of a test case
        pipeline (Pipeline): The pipeline that generated the answer
        question (str): The question for the test case
        answer (str): The answer of the pipeline for the test case
        score (int | float): The score of the LLM-as-judge for the test case
        label ('0' | "1"): The label of whether an answer is correct or a hallucination (using a threshold of 0.7)
    """
    pass

def run_test_cases(test_cases: list[dict]):
    """
    Runs all 100 test cases for the present metric / benchmark.

    Args:
        test_cases (list[dict]): The list of dicts returned by load_test_cases()
    """
    for t in test_cases:
        run_test_case(t)

def main():
    test_cases = load_test_cases()
    run_test_cases(test_cases)