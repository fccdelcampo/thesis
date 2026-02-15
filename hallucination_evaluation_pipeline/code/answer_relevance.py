import json
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import RAGAS and Config
from ragas import evaluate
from ragas.metrics import answer_relevancy
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
    Reads answer_relevance.jsonl or fallback.
    """
    file_path = os.path.join(TEST_CASES_DIR, "answer_relevance.jsonl")
    test_cases = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_cases.append(json.loads(line))
    
    # Fallback to truthfulqa 
    if not test_cases:
        tqa_path = os.path.join(TEST_CASES_DIR, "truthfulqa.jsonl")
        if os.path.exists(tqa_path):
            with open(tqa_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        test_cases.append(json.loads(line))
                        
    return test_cases

def evaluate_with_ragas(question: str, answer: str, contexts: list[str]) -> float:
    """
    Evaluates Answer Relevance using RAGAS.
    """
    if not answer:
        return 0.0

    # RAGAS expects a dataset
    data = {
        'question': [question],
        'answer': [answer],
        'contexts': [contexts],
        'ground_truth': [""] # Answer relevance doesn't strictly need ground truth
    }
    dataset = Dataset.from_dict(data)
    
    try:
        results = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy],
            llm=ragas_config.get_ragas_llm(),
            embeddings=ragas_config.get_ragas_embeddings()
        )
        return results['answer_relevancy']
    except Exception as e:
        print(f"RAGAS Evaluation Failed: {e}")
        return 0.0

def run_test_case(test_case: dict, test_case_id: int):
    question = test_case.get("question")
    if not question:
        return

    print(f"Running Answer Relevance {test_case_id}: {question[:50]}...")

    # 1. GraphRAG
    try:
        gr_result = run_graphrag(question)
        gr_answer = gr_result.get("answer", "")
        # Ensure contexts is a list of strings
        gr_contexts = gr_result.get("contexts", [])
        if isinstance(gr_contexts, str):
            gr_contexts = [gr_contexts]
            
        gr_score = evaluate_with_ragas(question, gr_answer, gr_contexts)
    except Exception as e:
        print(f"GraphRAG Failed: {e}")
        gr_score = 0.0

    # 2. Prolog GraphRAG
    try:
        pgr_result = run_prolog_graphrag(question, flag="prolog")
        pgr_answer = pgr_result.get("answer", "")
        pgr_contexts = pgr_result.get("contexts", [])
        if isinstance(pgr_contexts, str):
            pgr_contexts = [pgr_contexts]
            
        pgr_score = evaluate_with_ragas(question, pgr_answer, pgr_contexts)
    except Exception as e:
        print(f"Prolog GraphRAG Failed: {e}")
        pgr_score = 0.0

    # 3. Tuned (Placeholder)
    tuned_score = 0.0

    log_results(test_case_id, tuned_score, gr_score, pgr_score)

def log_results(test_case_id: int, tuned_score: float, graphrag_score: float, prolog_graphrag_score: float) -> None:
    file_path = os.path.join(RESULTS_DIR, "answer_relevance_scores.csv")
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
    print(f"Loaded {len(test_cases)} test cases for Answer Relevance.")
    for i, test_case in enumerate(tqdm(test_cases)):
        run_test_case(test_case, i)
        # pass

if __name__ == "__main__":
    main()