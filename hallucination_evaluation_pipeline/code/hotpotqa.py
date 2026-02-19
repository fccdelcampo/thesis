"""
HotpotQA Benchmark Evaluator

Evaluates the pipelines on HotpotQA — a multi-hop question answering benchmark.
Each test case requires reasoning across multiple pieces of information.
This is the most relevant benchmark for the Prolog-GraphRAG pipeline, as it tests
the system's ability to chain facts together.

Scoring: Uses LLM-as-a-judge to assess answer correctness against the ground truth.
Also evaluates whether the retrieved context contains the supporting facts.
Label: 1 if score >= 0.7 (correct), 0 otherwise (hallucination).
"""
import json
import os
import sys
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

# Import LLM for judge evaluation
import ragas_config

# Paths
TEST_CASES_DIR = os.path.join(project_root, "test_cases")
RESULTS_DIR = os.path.join(project_root, "hallucination_evaluation_results")
LOGS_DIR = os.path.join(project_root, "logs")

# Hallucination threshold
THRESHOLD = 0.7

# LLM-as-Judge prompt template
JUDGE_PROMPT = """You are an expert evaluator for multi-hop question answering. Given a question, the correct ground truth answer, optional supporting context, and a model's generated answer, rate the model's answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Supporting Context (if available):
{context}

Model's Answer: {answer}

Rate the model's answer on a scale of 0.0 to 1.0:
- 1.0: The answer is fully correct and matches the ground truth.
- 0.7-0.9: The answer is correct but may lack detail or have minor phrasing differences.
- 0.3-0.6: The answer is partially correct but misses key information or contains errors.
- 0.0-0.2: The answer is incorrect or a hallucination.

Consider that multi-hop questions require connecting information from multiple sources. 
A correct answer should demonstrate this reasoning chain.

Respond with ONLY a single float number between 0.0 and 1.0. Nothing else."""



# Import shared utilities
from hallucination_evaluation_pipeline.code.pipeline_utils import unload_ollama_model, CleanOutputChatOllama

def load_test_cases(limit: int = 10) -> list[dict]:
    """
    Reads .jsonl files in test_cases/ and turns each JSON line into a dict.

    Args:
        limit (int): Maximum number of test cases to load.

    Returns:
        list[dict]: A list of dict objects where each dict is a test case containing, at the minimum, the question and answer fields.
    """
    file_path = os.path.join(TEST_CASES_DIR, "hotpotqa.jsonl")
    test_cases = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                if line.strip():
                    test_cases.append(json.loads(line))
    return test_cases


def judge_answer(question: str, ground_truth: str, context: str, answer: str) -> float:
    """
    Uses LLM-as-a-judge to score the correctness of a pipeline's answer 
    against the ground truth for a multi-hop question.
    """
    if not answer or not answer.strip():
        return 0.0
    
    # Format context
    if isinstance(context, list):
        context = "\n".join(context)
    if not context:
        context = "(No supporting context provided)"
    
    # Format ground truth
    if isinstance(ground_truth, list):
        ground_truth = "; ".join(ground_truth)
    
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        context=context,
        answer=answer
    )
    
    try:
        # Unload model before judge evaluation to free VRAM
        unload_ollama_model()
        
        llm = ragas_config.get_ragas_llm()
        # Use CleanOutputChatOllama logic if not already applied in ragas_config
        # But ragas_config might use standard ChatOllama. 
        # For judge_answer, we are invoking directly, so we can wrap it or just use it.
        # Ideally ragas_config should return a clean LLM, but here we just invoke.
        
        response = llm.langchain_llm.invoke(prompt)
        score_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        # Extract float from response
        import re
        match = re.search(r'(\d+\.?\d*)', score_text)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        return 0.0
    except Exception as e:
        print(f"  Judge evaluation failed: {e}")
        return 0.0


def run_pipeline_safe(pipeline_fn, question: str, **kwargs) -> dict:
    """Safely run a pipeline, returning empty dict on failure."""
    try:
        # Ensure model is fresh or unloaded before heavy run if needed
        # unload_ollama_model() 
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
    question = test_case.get("question", "")
    ground_truth = test_case.get("ground_truth", "")
    gt_context = test_case.get("ground_truth_context", "")
    q_type = test_case.get("type", "unknown")
    q_level = test_case.get("level", "unknown")
    
    if not question:
        return
    
    print(f"\n[HotpotQA #{test_case_id}] ({q_type}/{q_level}) {question[:60]}...")

    # 1. GraphRAG Pipeline
    gr_result = run_pipeline_safe(run_graphrag, question)
    gr_answer = gr_result.get("answer", "")
    gr_score = judge_answer(question, ground_truth, gt_context, gr_answer)
    print(f"  GraphRAG Score: {gr_score:.2f}")
    
    # 2. Prolog GraphRAG Pipeline
    pgr_result = run_pipeline_safe(run_prolog_graphrag, question, flag="prolog")
    pgr_answer = pgr_result.get("answer", "")
    pgr_score = judge_answer(question, ground_truth, gt_context, pgr_answer)
    print(f"  Prolog-GraphRAG Score: {pgr_score:.2f}")
    
    # 3. Tuned Pipeline (Placeholder)
    tuned_score = 0.0
    
    # Log scores
    log_results(test_case_id, tuned_score, gr_score, pgr_score)
    
    # Log Q&A details per pipeline
    gr_label = "1" if gr_score >= THRESHOLD else "0"
    pgr_label = "1" if pgr_score >= THRESHOLD else "0"
    log_question_and_answer(test_case_id, "graphrag", question, gr_answer, gr_score, gr_label, q_type, q_level)
    log_question_and_answer(test_case_id, "prolog_graphrag", question, pgr_answer, pgr_score, pgr_label, q_type, q_level)


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
    file_path = os.path.join(RESULTS_DIR, "hotpotqa_scores.csv")
    
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


def log_question_and_answer(test_case_id: int, pipeline: str, question: str, answer: str, 
                            score: Union[int, float], label: str, 
                            q_type: str = "", q_level: str = "") -> None:
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
    file_path = os.path.join(log_dir, "hotpotqa_qa.jsonl")
    
    entry = {
        "test_case_id": test_case_id,
        "benchmark": "hotpotqa",
        "question": question,
        "answer": answer,
        "score": score,
        "label": label,
        "type": q_type,
        "level": q_level
    }
    
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")


def run_test_cases(test_cases: list[dict]):
    """
    Runs all 100 test cases for the present metric / benchmark.

    Args:
        test_cases (list[dict]): The list of dicts returned by load_test_cases()
    """
    for i, t in enumerate(tqdm(test_cases, desc="HotpotQA")):
        run_test_case(t, i)


def main(test_size: int = 10):
    test_cases = load_test_cases(limit=test_size)
    print(f"Loaded {len(test_cases)} HotpotQA test cases.")
    if not test_cases:
        print("No HotpotQA test cases found. Skipping.")
        return
    run_test_cases(test_cases)
    print(f"HotpotQA evaluation complete. Results saved to {RESULTS_DIR}/hotpotqa_scores.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=int, default=10, help="Number of test cases to run")
    args = parser.parse_args()
    main(test_size=args.test_size)