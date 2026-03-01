"""
AI2 ARC Benchmark Evaluator

Evaluates the pipelines on AI2 ARC (AI2 Reasoning Challenge).
AI2 ARC is a multiple-choice benchmark. Each test case has a question, 
choices (typically A-D), and a correct answer letter. We evaluate by checking 
if the pipeline's answer matches the correct letter.

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


def load_test_cases(limit: int = 1000, start_idx: int = 0) -> list[dict]:
    """
    Reads .jsonl files in test_cases/ and turns each JSON line into a dict.

    Args:
        limit (int): Max number of test cases.
        start_idx (int): Global offset to skip initial test cases.

    Returns:
        list[dict]: A list of dict objects where each dict is a test case.
    """
    file_path = os.path.join(TEST_CASES_DIR, "ai2_arc.jsonl")
    test_cases = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start_idx:
                    continue
                if i >= start_idx + limit:
                    break
                if line.strip():
                    test_cases.append(json.loads(line))
    return test_cases


def extract_answer_letter(response: str) -> str:
    """
    Extracts the answer letter (A, B, C, or D) from the pipeline's response.
    Tries multiple patterns: standalone letter, "Answer: X", "The answer is X", etc.
    """
    if not response:
        return ""
    
    response = response.strip()
    
    # Pattern 1: Explicit "Final Answer: X" or "The correct answer is X" anywhere
    match = re.search(r'(?:final\s+answer|correct\s+answer\s+is|answer\s+is)[\s\*:-]*([A-D])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # Pattern 2: "Answer: X" or "Choice: X" anywhere
    match = re.search(r'(?:answer|choice)[\s\*:-]*([A-D])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Standalone letter at the VERY END of the response
    match = re.search(r'(?:\*\*|\*|\s|^)([A-D])(?:\*\*|\*|\s|\.|\)|$)', response[-10:], re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 4: Standalone letter at the VERY START (fallback for old outputs)
    match = re.match(r'^[\*\s]*([A-D])(?:[\.\)]|\b)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
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
        return {"answer": "", "contexts": [], "text_answer": ""}
    

def run_test_case(test_case: dict, test_case_id: int):
    """
    Runs a test case by calling each pipeline's run_pipeline function. 
    """
    question = test_case.get("formatted_input", "")
    correct_answer = test_case.get("correct_answer", "")
    
    print(f"\n" + "="*80)
    print(f"[AI2 ARC #{test_case_id}] {question[:120]}...")
    print(f"Correct Answer: [{correct_answer}]")
    print("-" * 80)

    # 1. GraphRAG Pipeline
    gr_result = run_pipeline_safe(run_graphrag, question)
    gr_letter = extract_answer_letter(gr_result.get("answer", ""))
    gr_score = score_answer(gr_letter, correct_answer)
    
    # 2. Prolog GraphRAG Pipeline
    pgr_result = run_pipeline_safe(run_prolog_graphrag, question, flag="prolog")
    pgr_letter = extract_answer_letter(pgr_result.get("answer", ""))
    pgr_score = score_answer(pgr_letter, correct_answer)
    
    # 3. Tuned Pipeline 
    tuned_result = run_pipeline_safe(run_tuned_pipeline, question)
    tuned_letter = extract_answer_letter(tuned_result.get("answer", ""))
    tuned_score = score_answer(tuned_letter, correct_answer)
    
    print(f"\n--- PIPELINE COMPARISON ---")
    def format_res(name, letter, score):
        status = "CORRECT" if score >= 1.0 else "WRONG"
        let_str = f"[{letter}]" if letter else "[?]"
        return f"{name:<15} | Predicted: {let_str:<5} | {status}"
        
    print(format_res("GraphRAG", gr_letter, gr_score))
    print(format_res("Prolog", pgr_letter, pgr_score))
    print(format_res("Tuned LLM", tuned_letter, tuned_score))
    print("=" * 80 + "\n")
    
    # Log scores
    log_results(test_case_id, tuned_score, gr_score, pgr_score)
    
    # Log Q&A details per pipeline
    gr_label = "0" if gr_score >= THRESHOLD else "1"
    pgr_label = "0" if pgr_score >= THRESHOLD else "1"
    tuned_label = "0" if tuned_score >= THRESHOLD else "1"
    
    log_question_and_answer(test_case_id, "graphrag", question, gr_result.get("answer", ""), gr_score, gr_label)
    log_question_and_answer(test_case_id, "prolog_graphrag", question, pgr_result.get("answer", ""), pgr_score, pgr_label)
    log_question_and_answer(test_case_id, "tuned_pipeline", question, tuned_result.get("answer", ""), tuned_score, tuned_label)


def log_results(test_case_id: int, tuned_score: Union[float, int], 
                graphrag_score: Union[float, int], prolog_graphrag_score: Union[float, int]) -> None:
    """
    Logs results of the evaluations in the appropriate CSV files in hallucination_evaluation_results/
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = os.path.join(RESULTS_DIR, "ai2_arc_scores.csv")
    
    new_row = pd.DataFrame({
        "test_case_id": [test_case_id],
        "tuned_score": [tuned_score],
        "tuned_label": [0 if tuned_score >= THRESHOLD else 1],
        "graphrag_score": [graphrag_score],
        "graphrag_label": [0 if graphrag_score >= THRESHOLD else 1],
        "prolog_graphrag_score": [prolog_graphrag_score],
        "prolog_graphrag_label": [0 if prolog_graphrag_score >= THRESHOLD else 1]
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
    file_path = os.path.join(log_dir, "ai2_arc_qa.jsonl")
    
    entry = {
        "test_case_id": test_case_id,
        "benchmark": "ai2_arc",
        "question": question,
        "answer": answer,
        "score": score,
        "label": label
    }
    
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")


def run_test_cases(test_cases: list[dict], start_idx: int = 0):
    for i, t in enumerate(tqdm(test_cases, desc="AI2 ARC"), start=start_idx):
        run_test_case(t, i)


def main(test_size: int = 1000, start_idx: int = 0):
    test_cases = load_test_cases(limit=test_size, start_idx=start_idx)
    print(f"Loaded {len(test_cases)} AI2 ARC test cases (starting from index {start_idx}).")
    if not test_cases:
        print("No AI2 ARC test cases found. Skipping.")
        return
    run_test_cases(test_cases, start_idx=start_idx)
    print(f"AI2 ARC evaluation complete. Results saved to {RESULTS_DIR}/ai2_arc_scores.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=int, default=1000, help="Number of test cases to run")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index")
    args = parser.parse_args()
    main(test_size=args.test_size, start_idx=args.start_idx)
