import json
import os
import sys
import pandas as pd
from tqdm import tqdm
import time
from typing import Tuple

# Add project root to sys.path to allow importing from parallel directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import RAGAS and Config
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
from . import ragas_config

from prolog_graphrag_pipeline.llm import generate as llm_generate
from prolog_graphrag_pipeline.config import LLM_SYSTEM_PROMPT

# Define paths
# test_cases/ is two levels up from this file
TEST_CASES_DIR = os.path.join(project_root, "test_cases")
RESULTS_DIR = os.path.join(project_root, "hallucination_evaluation_results/data")

CORRECTNESS_THRESHOLD = 0.7

def load_test_cases() -> list[dict]:
    """
    Reads .jsonl files in test_cases/ and turns each JSON line into a dict.
    Specifically targets faithfulness.jsonl (or creates synthetic ones if empty).
    For now, we will use mmlu.jsonl as a proxy if faithfulness.jsonl is empty, 
    or just return an empty list if strict. The plan says use faithfulness.jsonl.
    """
    file_path = os.path.join(TEST_CASES_DIR, "metrics_test_cases.jsonl")
    test_cases = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_cases.append(json.loads(line))
    return test_cases


def run_pipeline_safe(pipeline, pipeline_fn, question: str, **kwargs) -> dict:
    """Safely run a pipeline, returning empty dict on failure. Injects 'elapsed' key."""
    import time
    t0 = time.perf_counter()
    try:
        result = pipeline_fn(question, **kwargs)
    except Exception as e:
        print(f"  Pipeline error in {pipeline} pipeline: {e}")
        result = {"answer": "", "contexts": [], "text_answer": ""}
    result["elapsed"] = time.perf_counter() - t0
    return result


def evaluate_with_ragas(question: str, answer: str, contexts: list[str], ground_truth: str) -> Tuple[float, float, float]:
    """
    Evaluates Faithfulness, Answer Relevancy, and Context Recall using RAGAS.
    """
    if not answer or not contexts:
        return 0.0

    # RAGAS expects a dataset
    data = {
        'question': [question],
        'answer': [answer],
        'contexts': [contexts],
        'ground_truth': [ground_truth] # Faithfulness doesn't strictly need ground truth
    }
    dataset = Dataset.from_dict(data)
    
    try:
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_recall, answer_relevancy],
            llm=ragas_config.get_ragas_llm(),
            embeddings=ragas_config.get_ragas_embeddings()
        )
        
        return results['faithfulness'].iloc[0], results['context_recall'].iloc[0], results['answer_relevancy'].iloc[0]
    except Exception as e:
        print(f"RAGAS Evaluation Failed: {e}")
        return 0.0, 0.0, 0.0
    
import pandas as pd
from datasets import Dataset
from typing import List

def batch_evaluate_with_ragas(
    questions: List[str], 
    answers: List[str], 
    contexts: List[List[str]], 
    ground_truths: List[str]
) -> pd.DataFrame:
    """
    Evaluates a batch of RAG results and returns a DataFrame with scores per row.
    """
    
    # 1. Validation: Ensure all lists are the same length
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("All input lists must have the same length.")

    # 2. Construct the Dataset
    # Note: 'contexts' must be a list of lists of strings
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    try:
        # 3. Run Batch Evaluation
        # Ragas handles the iteration and concurrency internally
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_recall, answer_relevancy],
            llm=ragas_config.get_ragas_llm(),
            embeddings=ragas_config.get_ragas_embeddings()
        )
        
        # 4. Convert results to a pandas DataFrame for easy inspection
        return results.to_pandas()

    except Exception as e:
        print(f"Batch RAGAS Evaluation Failed: {e}")
        # Return an empty DataFrame or one filled with 0.0/NaN depending on your preference
        return pd.DataFrame()

# GraphRAG
from graphrag_pipeline.graphrag.graphrag_driver import run_pipeline as run_graphrag

# Prolog GraphRAG
# Import as package from project root
from prolog_graphrag_pipeline.main_driver import run_pipeline as run_prolog_graphrag

# Tuned Pipeline (Assuming it exists or we skip)
from tuned_pipeline.main_driver import run_pipeline as run_tuned

def run_test_case(test_case: dict, test_case_id: int):
    """
    Runs a test case by calling each pipeline's run_pipeline function.
    """
    question = test_case.get("question")
    ground_truth = test_case.get("answer")
    if not question:
        return

    print(f"Running Test Case {test_case_id}: {question[:50]}...")

    # 1. Run GraphRAG Pipeline
    try:
        # Expected return: dict with "answer", "context"
        graphrag_result = run_pipeline_safe(pipeline="graphrag", pipeline_fn=run_graphrag, question=question)
        graphrag_answer = graphrag_result.get("answer", "")
        # Ensure contexts is a list of strings
        graphrag_context = graphrag_result.get("context", []) or graphrag_result.get("contexts", [])
        if isinstance(graphrag_context, str):
            graphrag_context = [graphrag_context]
        
        # Evaluate Faithfulness
        faithfulness_gr_score, context_recall_gr_score, answer_relevancy_gr_score = evaluate_with_ragas(question, graphrag_answer, graphrag_context, ground_truth)
    except Exception as e:
        print(f"GraphRAG Failed: {e}")
        faithfulness_gr_score = 0.0
        context_recall_gr_score = 0.0
        answer_relevancy_gr_score = 0.0
    
    # 2. Run Prolog GraphRAG Pipeline
    try:
        # Expected return: dict with "answer", "contexts" (list)
        # run_pipeline(question: str, flag: str, sample_mode: bool = False, include_conclusion: bool = True)
        prolog_result = run_pipeline_safe(pipeline="prolog_graphrag", pipeline_fn=run_prolog_graphrag, question=question, flag="x")
        prolog_answer = prolog_result.get("answer", "")
        # content is likely a list or string
        contexts = prolog_result.get("contexts", [])
        if isinstance(contexts, str):
            contexts = [contexts]
        
        # Evaluate Faithfulness
        faithfulness_pgr_score, context_recall_pgr_score, answer_relevancy_pgr_score = evaluate_with_ragas(question, prolog_answer, contexts, ground_truth)
    except Exception as e:
        print(f"Prolog GraphRAG Failed: {e}")
        faithfulness_pgr_score = 0.0
        context_recall_pgr_score = 0.0
        answer_relevancy_pgr_score = 0.0
    
    # Log Results
    log_results(test_case_id, faithfulness_gr_score, faithfulness_pgr_score, "faithfulness")
    log_results(test_case_id, context_recall_gr_score, context_recall_pgr_score, "context_recall")
    log_results(test_case_id, answer_relevancy_gr_score, answer_relevancy_pgr_score, "answer_relevancy")
    
    
from typing import List, Dict
import pandas as pd

def run_test_case_batch(test_cases: List[Dict]):
    """
    Runs a batch of test cases for both GraphRAG and Prolog GraphRAG,
    then evaluates them in bulk using RAGAS.
    """
    # Storage for pipeline outputs
    gr_data = {"questions": [], "answers": [], "contexts": [], "ground_truths": []}
    pgr_data = {"questions": [], "answers": [], "contexts": [], "ground_truths": []}
    test_case_ids = []

    # 1. Execute all pipeline runs first
    for i, test_case in enumerate(test_cases):
        question = test_case.get("question")
        ground_truth = test_case.get("answer", "")
        if not question:
            continue
        
        test_case_ids.append(i)
        print(f"Processing Pipeline Runs for Test Case {i}: {question[:50]}...")

        # --- GraphRAG Run ---
        try:
            gr_res = run_graphrag(question)
            gr_ans = gr_res.get("answer", "")
            gr_ctx = gr_res.get("context", []) or gr_res.get("contexts", [])
            gr_ctx = [gr_ctx] if isinstance(gr_ctx, str) else gr_ctx
            
            gr_data["questions"].append(question)
            gr_data["answers"].append(gr_ans)
            gr_data["contexts"].append(gr_ctx)
            gr_data["ground_truths"].append(ground_truth)
        except Exception as e:
            print(f"GraphRAG Execution failed for case {i}: {e}")

        # --- Prolog GraphRAG Run ---
        try:
            pgr_res = run_prolog_graphrag(question)
            pgr_ans = pgr_res.get("answer", "")
            pgr_ctx = pgr_res.get("contexts", [])
            pgr_ctx = [pgr_ctx] if isinstance(pgr_ctx, str) else pgr_ctx
            
            pgr_data["questions"].append(question)
            pgr_data["answers"].append(pgr_ans)
            pgr_data["contexts"].append(pgr_ctx)
            pgr_data["ground_truths"].append(ground_truth)
        except Exception as e:
            print(f"Prolog GraphRAG Execution failed for case {i}: {e}")

    # 2. Perform Batch Evaluation
    print("\nStarting Batch RAGAS Evaluation...")
    
    df_gr = batch_evaluate_with_ragas(
        gr_data["questions"], gr_data["answers"], gr_data["contexts"], gr_data["ground_truths"]
    )
    
    df_pgr = batch_evaluate_with_ragas(
        pgr_data["questions"], pgr_data["answers"], pgr_data["contexts"], pgr_data["ground_truths"]
    )

    # 3. Log Results per test case
    # Assuming the order in the resulting DataFrame matches the input order
    for idx, t_id in enumerate(test_case_ids):
        # Log GraphRAG vs Prolog GraphRAG for each metric
        for metric in ["faithfulness", "context_recall", "answer_relevancy"]:
            score_gr = df_gr.iloc[idx][metric] if not df_gr.empty else 0.0
            score_pgr = df_pgr.iloc[idx][metric] if not df_pgr.empty else 0.0
            log_results(t_id, score_gr, score_pgr, metric)

    return df_gr, df_pgr


def log_results(test_case_id: int, graphrag_score: float, prolog_graphrag_score: float, metric: str) -> None:
    if metric == "faithfulness":
        file_path = os.path.join(RESULTS_DIR, "faithfulness_scores.csv")
    elif metric == "context_recall":
        file_path = os.path.join(RESULTS_DIR, "context_recall_scores.csv")
    elif metric == "answer_relevancy":
        file_path = os.path.join(RESULTS_DIR, "answer_relevancy_scores.csv")
    else:
        raise ValueError(f"Unknown metric: {metric}")
    new_row = pd.DataFrame({
        "test_case_id": [test_case_id],
        "graphrag_score": [graphrag_score],
        "graphrag_label": [0 if graphrag_score >= CORRECTNESS_THRESHOLD else 1],
        "prolog_graphrag_score": [prolog_graphrag_score],
        "prolog_graphrag_label": [0 if prolog_graphrag_score >= CORRECTNESS_THRESHOLD else 1]
    })
    
    header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
    new_row.to_csv(file_path, mode='a', header=header, index=False)

def main():
    test_cases = load_test_cases()
    run_test_case_batch(test_cases)
    # print(f"Loaded {len(test_cases)} test cases for Faithfulness.")
    
    # for i, test_case in enumerate(tqdm(test_cases)):
    #     run_test_case(test_case, i)

if __name__ == "__main__":
    main()