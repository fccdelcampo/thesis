"""
MMLU Benchmark Evaluator

Evaluates the pipelines on MMLU (Massive Multitask Language Understanding).
MMLU is a multiple-choice benchmark. Each test case has a question, 4 options (A-D),
and a correct answer letter. We evaluate by checking if the pipeline's answer
matches the correct letter.

Scoring: 1.0 if correct letter matches, 0.0 otherwise.
Label: 1 if score >= 0.7 (i.e., correct), 0 otherwise (hallucination/incorrect).
"""
import json
import os
import sys
import re
import pandas as pd
from tqdm import tqdm
from typing import Union

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import pipeline drivers
from graphrag_pipeline.graphrag.graphrag_driver import run_pipeline as run_graphrag
from prolog_graphrag_pipeline.main_driver import run_pipeline as run_prolog_graphrag

# Paths
TEST_CASES_DIR = os.path.join(project_root, "test_cases")
RESULTS_DIR = os.path.join(project_root, "hallucination_evaluation_results")
LOGS_DIR = os.path.join(project_root, "logs")

# Hallucination threshold
THRESHOLD = 0.7


def load_test_cases() -> list[dict]:
    """
    Reads .jsonl files in test_cases/ and turns each JSON line into a dict.

    Args:
        None. Path must be defined in the function itself.

    Returns:
        list[dict]: A list of dict objects where each dict is a test case containing, at the minimum, the question and answer fields.
    """
    file_path = os.path.join(TEST_CASES_DIR, "mmlu.jsonl")
    test_cases = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_cases.append(json.loads(line))
    return test_cases


def format_mmlu_question(test_case: dict) -> str:
    """
    Formats an MMLU test case into a prompt string with options.
    Example: "Question: ... \nA. ... \nB. ... \nC. ... \nD. ...\nAnswer with the letter only."
    """
    question = test_case.get("question", "")
    options = test_case.get("options", [])
    letters = ["A", "B", "C", "D"]
    
    formatted = f"{question}\n\n"
    for i, opt in enumerate(options):
        if i < len(letters):
            formatted += f"{letters[i]}. {opt}\n"
    formatted += "\nAnswer with the letter only (A, B, C, or D)."
    return formatted


def extract_answer_letter(response: str) -> str:
    """
    Extracts the answer letter (A, B, C, or D) from the pipeline's response.
    Tries multiple patterns: standalone letter, "Answer: X", "The answer is X", etc.
    """
    if not response:
        return ""
    
    response = response.strip()
    
    # Pattern 1: Standalone letter at the start
    match = re.match(r'^([A-D])\b', response.upper())
    if match:
        return match.group(1)
    
    # Pattern 2: "Answer: X" or "The answer is X"
    match = re.search(r'(?:answer|choice)[\s:is]*([A-D])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: Any standalone A-D in the response
    match = re.search(r'\b([A-D])\b', response.upper())
    if match:
        return match.group(1)
    
    return ""


def score_answer(predicted: str, correct: str) -> float:
    """Returns 1.0 if the predicted letter matches the correct letter, else 0.0."""
    return 1.0 if predicted.upper() == correct.upper() else 0.0


def run_pipeline_safe(pipeline_fn, question: str, **kwargs) -> dict:
    """Safely run a pipeline, returning empty dict on failure."""
    try:
        return pipeline_fn(question, **kwargs)
    except Exception as e:
        print(f"  Pipeline error: {e}")
        return {"answer": "", "contexts": []}
    


def run_test_case(test_case: dict, test_case_id: int):
    """
    Runs a test case by calling each pipeline's run_pipeline function. 
    Calls necessary LLM-as-judge evaluations after pipeline returns answer dict.
    Calls log_question_and_answer() to log the question and answer for each test case.
    Calls log_results() after all evaluations have returned scores

    Args:
        test_case (dict): A test case's dict
        test_case_id (int): The ID of the test case
    """
    formatted_question = format_mmlu_question(test_case)
    correct_answer = test_case.get("answer", "")
    raw_question = test_case.get("question", "")
    
    print(f"\n[MMLU #{test_case_id}] {raw_question[:60]}...")
    print(f"  Correct: {correct_answer}")

    # 1. GraphRAG Pipeline
    gr_result = run_pipeline_safe(run_graphrag, formatted_question)
    gr_letter = extract_answer_letter(gr_result.get("answer", ""))
    gr_score = score_answer(gr_letter, correct_answer)
    print(f"  GraphRAG: {gr_letter} (Score: {gr_score})")
    
    # 2. Prolog GraphRAG Pipeline
    pgr_result = run_pipeline_safe(run_prolog_graphrag, formatted_question, flag="prolog")
    pgr_letter = extract_answer_letter(pgr_result.get("answer", ""))
    pgr_score = score_answer(pgr_letter, correct_answer)
    print(f"  Prolog-GraphRAG: {pgr_letter} (Score: {pgr_score})")
    
    # 3. Tuned Pipeline (Placeholder — uses direct LLM without RAG)
    tuned_score = 0.0
    
    # Log scores
    log_results(test_case_id, tuned_score, gr_score, pgr_score)
    
    # Log Q&A details per pipeline
    gr_label = "1" if gr_score >= THRESHOLD else "0"
    pgr_label = "1" if pgr_score >= THRESHOLD else "0"
    log_question_and_answer(test_case_id, "graphrag", raw_question, gr_result.get("answer", ""), gr_score, gr_label)
    log_question_and_answer(test_case_id, "prolog_graphrag", raw_question, pgr_result.get("answer", ""), pgr_score, pgr_label)


def log_results(test_case_id: int, tuned_score: Union[float, int], 
                graphrag_score: Union[float, int], prolog_graphrag_score: Union[float, int]) -> None:
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = os.path.join(RESULTS_DIR, "mmlu_scores.csv")
    
    new_row = pd.DataFrame({
        "test_case_id": [test_case_id],
        "tuned_score": [tuned_score],
        "tuned_label": [1 if tuned_score >= THRESHOLD else 0],
        "graphrag_score": [graphrag_score],
        "graphrag_label": [1 if graphrag_score >= THRESHOLD else 0],
        "prolog_graphrag_score": [prolog_graphrag_score],
        "prolog_graphrag_label": [1 if prolog_graphrag_score >= THRESHOLD else 0]
    })
    
    header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
    new_row.to_csv(file_path, mode='a', header=header, index=False)


def log_question_and_answer(test_case_id: int, pipeline: str, question: str, 
                            answer: str, score: Union[int, float], label: str) -> None:
    """
    Logs the question and answer for a given test case ID to a CSV file in logs/{pipeline}_logs/.
    Each log is a JSON line.

    Args:
        test_case_id (int): The ID of a test case
        pipeline (str): The pipeline that generated the answer
        question (str): The question for the test case
        answer (str): The answer of the pipeline for the test case
        score (int | float): The score of the LLM-as-judge for the test case
        label ('0' | "1"): The label of whether an answer is correct or a hallucination (using a threshold of 0.7)
    """
    log_dir = os.path.join(LOGS_DIR, f"{pipeline}_logs")
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "mmlu_qa.jsonl")
    
    entry = {
        "test_case_id": test_case_id,
        "benchmark": "mmlu",
        "question": question,
        "answer": answer,
        "score": score,
        "label": label
    }
    
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")


def run_test_cases(test_cases: list[dict]):
    """
    Runs all 100 test cases for the present metric / benchmark.

    Args:
        test_cases (list[dict]): The list of dicts returned by load_test_cases()
    """
    for i, t in enumerate(tqdm(test_cases, desc="MMLU")):
        run_test_case(t, i)


def main():
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} MMLU test cases.")
    if not test_cases:
        print("No MMLU test cases found. Skipping.")
        return
    run_test_cases(test_cases)
    print(f"MMLU evaluation complete. Results saved to {RESULTS_DIR}/mmlu_scores.csv")


if __name__ == "__main__":
    main()