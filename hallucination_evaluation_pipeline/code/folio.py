"""
FOLIO Benchmark Evaluator

Evaluates the pipelines on FOLIO (First-Order Logic Inference Dataset).
Each test case provides a series of premises and asks what can be concluded.
The answer is categorical (e.g., "True", "False", "Unknown").
We evaluate by checking if the pipeline's answer extracts the correct category.

Scoring: 1.0 if correct label matches, 0.0 otherwise.
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
try:
    from tuned_pipeline.main_driver import run_pipeline as run_tuned_pipeline
except ImportError:
    print("Warning: Could not import tuned_pipeline. Check sys.path and dependencies.")
    def run_tuned_pipeline(q): return {"text_answer": "Error: Pipeline not found"}


# Paths
TEST_CASES_DIR = os.path.join(project_root, "test_cases")
RESULTS_DIR = os.path.join(project_root, "hallucination_evaluation_results")
LOGS_DIR = os.path.join(project_root, "logs")

# Hallucination threshold
THRESHOLD = 0.7


def load_test_cases(limit: int = 1000) -> list[dict]:
    """
    Reads .jsonl files in test_cases/ and turns each JSON line into a dict.
    """
    file_path = os.path.join(TEST_CASES_DIR, "folio.jsonl")
    test_cases = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                if line.strip():
                    test_cases.append(json.loads(line))
    return test_cases


def extract_answer(response: str) -> str:
    """
    Extracts the FOLIO answer (True, False, or Unknown) from the pipeline's response.
    """
    if not response:
        return ""
    
    response = response.strip().lower()
    
    if "true" in response:
        return "True"
    elif "false" in response:
        return "False"
    elif "unknown" in response:
        return "Unknown"
    
    return ""


def score_answer(predicted: str, correct: str) -> float:
    """Returns 1.0 if the predicted matches the correct label, else 0.0."""
    return 1.0 if predicted.lower() == correct.lower() else 0.0


def run_pipeline_safe(pipeline_fn, question: str, **kwargs) -> dict:
    """Safely run a pipeline, returning empty dict on failure."""
    try:
        return pipeline_fn(question, **kwargs)
    except Exception as e:
        print(f"  Pipeline error: {e}")
        return {"answer": "", "contexts": [], "text_answer": ""}
    

def run_test_case(test_case: dict, test_case_id: int):
    """
    Runs a test case by calling each pipeline's run_pipeline function. 
    """
    # Build formatted question incorporating the premises
    raw_question = test_case.get("question", "")
    premises = test_case.get("ground_truth_context", "")
    
    formatted_question = f"{raw_question}\nPremises:\n{premises}\n\nAnswer with True, False, or Unknown."
    correct_answer = test_case.get("label", "")
    
    print(f"\n[FOLIO #{test_case_id}] {raw_question[:60]}...")
    print(f"  Correct: {correct_answer}")

    # 1. GraphRAG Pipeline
    gr_result = run_pipeline_safe(run_graphrag, formatted_question)
    gr_ans = extract_answer(gr_result.get("answer", ""))
    gr_score = score_answer(gr_ans, correct_answer)
    print(f"  GraphRAG: {gr_ans} (Score: {gr_score})")
    
    # 2. Prolog GraphRAG Pipeline
    pgr_result = run_pipeline_safe(run_prolog_graphrag, formatted_question, flag="prolog")
    pgr_ans = extract_answer(pgr_result.get("answer", ""))
    pgr_score = score_answer(pgr_ans, correct_answer)
    print(f"  Prolog-GraphRAG: {pgr_ans} (Score: {pgr_score})")
    
    # 3. Tuned Pipeline 
    tuned_result = run_pipeline_safe(run_tuned_pipeline, formatted_question)
    tuned_ans = extract_answer(tuned_result.get("text_answer", ""))
    tuned_score = score_answer(tuned_ans, correct_answer)
    print(f"  Tuned Pipeline: {tuned_ans} (Score: {tuned_score})")
    
    # Log scores
    log_results(test_case_id, tuned_score, gr_score, pgr_score)
    
    # Log Q&A details per pipeline
    gr_label = "1" if gr_score >= THRESHOLD else "0"
    pgr_label = "1" if pgr_score >= THRESHOLD else "0"
    tuned_label = "1" if tuned_score >= THRESHOLD else "0"
    
    log_question_and_answer(test_case_id, "graphrag", raw_question, gr_result.get("answer", ""), gr_score, gr_label)
    log_question_and_answer(test_case_id, "prolog_graphrag", raw_question, pgr_result.get("answer", ""), pgr_score, pgr_label)
    log_question_and_answer(test_case_id, "tuned_pipeline", raw_question, tuned_result.get("text_answer", ""), tuned_score, tuned_label)


def log_results(test_case_id: int, tuned_score: Union[float, int], 
                graphrag_score: Union[float, int], prolog_graphrag_score: Union[float, int]) -> None:
    """
    Logs results of the evaluations in the appropriate CSV files in hallucination_evaluation_results/
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = os.path.join(RESULTS_DIR, "folio_scores.csv")
    
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
    """
    log_dir = os.path.join(LOGS_DIR, f"{pipeline}_logs")
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "folio_qa.jsonl")
    
    entry = {
        "test_case_id": test_case_id,
        "benchmark": "folio",
        "question": question,
        "answer": answer,
        "score": score,
        "label": label
    }
    
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")


def run_test_cases(test_cases: list[dict]):
    for i, t in enumerate(tqdm(test_cases, desc="FOLIO")):
        run_test_case(t, i)


def main(test_size: int = 1000):
    test_cases = load_test_cases(limit=test_size)
    print(f"Loaded {len(test_cases)} FOLIO test cases.")
    if not test_cases:
        print("No FOLIO test cases found. Skipping.")
        return
    run_test_cases(test_cases)
    print(f"FOLIO evaluation complete. Results saved to {RESULTS_DIR}/folio_scores.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=int, default=1000, help="Number of test cases to run")
    args = parser.parse_args()
    main(test_size=args.test_size)
