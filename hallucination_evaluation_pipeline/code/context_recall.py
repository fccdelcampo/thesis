import json
import os
import sys
import pandas as pd
from tqdm import tqdm
import time
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import RAGAS and Config
from ragas import evaluate
from ragas.metrics import context_recall
from datasets import Dataset
import ragas_config

# Import drivers
# GraphRAG
from graphrag_pipeline.graphrag.graphrag_driver import run_pipeline as run_graphrag

# Prolog GraphRAG
# Import as package from project root
from prolog_graphrag_pipeline.main_driver import run_pipeline as run_prolog_graphrag

TEST_CASES_DIR = os.path.join(project_root, "test_cases")
RESULTS_DIR = os.path.join(project_root, "hallucination_evaluation_results")

def load_test_cases() -> list[dict]:
    """
    Reads context_recall.jsonl.
    If empty, we might need a fallback or just return empty.
    Context Recall requires 'ground_truth' to be present in the test case.
    """
    file_path = os.path.join(TEST_CASES_DIR, "context_recall.jsonl")
    test_cases = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_cases.append(json.loads(line))
    
    # Fallback to truthfulqa if empty (since it has ground_truth)
    if not test_cases:
        tqa_path = os.path.join(TEST_CASES_DIR, "truthfulqa.jsonl")
        if os.path.exists(tqa_path):
            with open(tqa_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        test_cases.append(json.loads(line))
                        
    return test_cases

def evaluate_with_ragas(question: str, ground_truth: str, contexts: list[str]) -> float:
    """
    Evaluates Context Recall using RAGAS.
    """
    if not ground_truth or not contexts:
        return 0.0
        
    # Handle list of ground truths (TruthfulQA style might have list)
    if isinstance(ground_truth, list):
        ground_truth = "; ".join(ground_truth)

    # RAGAS expects a dataset
    data = {
        'question': [question],
        'answer': ["placeholder"], # Context recall doesn't strictly need generated answer
        'contexts': [contexts],
        'ground_truth': [ground_truth]
    }
    dataset = Dataset.from_dict(data)
    
    try:
        start_time = time.time()
        results = evaluate(
            dataset=dataset,
            metrics=[context_recall],
            llm=ragas_config.get_ragas_llm(),
            embeddings=ragas_config.get_ragas_embeddings()
        )
        end_time = time.time()
        print(f"RAGAS Context Recall Evaluation took {end_time - start_time:.2f} seconds.")
        return results['context_recall']
    except Exception as e:
        print(f"RAGAS Evaluation Failed: {e}")
        return 0.0

def run_test_case(test_case: dict, test_case_id: int):
    question = test_case.get("question")
    ground_truth = test_case.get("ground_truth") or test_case.get("answer")
    
    if not question or not ground_truth:
        return

    print(f"Running Context Recall {test_case_id}: {question[:50]}...")

    # 1. GraphRAG
    try:
        gr_result = run_graphrag(question)
        # Ensure contexts is a list of strings
        gr_contexts = gr_result.get("contexts", []) or gr_result.get("context", [])
        if isinstance(gr_contexts, str):
            gr_contexts = [gr_contexts]
            
        gr_score = evaluate_with_ragas(question, ground_truth, gr_contexts)
    except Exception as e:
        print(f"GraphRAG Failed: {e}")
        gr_score = 0.0

    # 2. Prolog GraphRAG
    try:
        pgr_result = run_prolog_graphrag(question)
        contexts = pgr_result.get("contexts", [])
        if isinstance(contexts, str):
            contexts = [contexts]
            
        pgr_score = evaluate_with_ragas(question, ground_truth, contexts)
    except Exception as e:
        print(f"Prolog GraphRAG Failed: {e}")
        pgr_score = 0.0

    # 3. Tuned (Placeholder)
    tuned_score = 0.0

    log_results(test_case_id, tuned_score, gr_score, pgr_score)

def log_results(test_case_id: int, tuned_score: float, graphrag_score: float, prolog_graphrag_score: float) -> None:
    file_path = os.path.join(RESULTS_DIR, "context_recall_scores.csv")
    new_row = pd.DataFrame({
        "test_case_id": [test_case_id],
        "tuned_score": [tuned_score],
        "graphrag_score": [graphrag_score],
        "prolog_graphrag_score": [prolog_graphrag_score]
    })
    
    header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
    new_row.to_csv(file_path, mode='a', header=header, index=False)

def main():
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases for Context Recall.")
    for i, test_case in enumerate(tqdm(test_cases)):
        run_test_case(test_case, i)
        # pass

if __name__ == "__main__":
    main()