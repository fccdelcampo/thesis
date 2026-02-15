import json
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

def evaluate_context_recall(question: str, ground_truth: str, context: str) -> tuple[float, str]:
    """
    Evaluates if the ground truth answer can be attributed to the retrieved context.
    """
    if not context:
        return 0.0, "No context provided"
    
    # Handle list of ground truths (TruthfulQA style)
    if isinstance(ground_truth, list):
        ground_truth = "; ".join(ground_truth)

    prompt = f"""
    You are a judge evaluating the Context Recall of a RAG system.
    
    QUESTION: {question}
    GROUND TRUTH ANSWER: {ground_truth}
    RETRIEVED CONTEXT: {context}
    
    INSTRUCTION:
    1. Check if the RETRIEVED CONTEXT information is sufficient to derive the GROUND TRUTH ANSWER.
    2. Ignore extra information in the context.
    3. If the context contains the necessary information to answer the question as per the ground truth, score 1.
    4. If the context is missing key information, score 0.
    
    OUTPUT FORMAT:
    Return ONLY a JSON object with keys "score" (0 or 1) and "reason".
    Example: {{"score": 1, "reason": "Context contains all necessary facts."}}
    """
    
    try:
        import ollama
        response = ollama.chat(
            model='llama3:8b-instruct-fp16',
            messages=[{'role': 'system', 'content': 'You are a strict evaluator.'}, 
                      {'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )
        content = response['message']['content']
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = content[start:end]
            data = json.loads(json_str)
            return float(data.get("score", 0)), data.get("reason", "Parsed")
        else:
            return 0.0, "Failed to parse JSON"
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return 0.0, "Error"

def run_test_case(test_case: dict, test_case_id: int):
    question = test_case.get("question")
    ground_truth = test_case.get("ground_truth") or test_case.get("answer")
    
    if not question or not ground_truth:
        return

    print(f"Running Context Recall {test_case_id}: {question[:50]}...")

    # 1. GraphRAG
    try:
        gr_result = run_graphrag(question)
        gr_context = gr_result.get("context", "")
        gr_score, gr_reason = evaluate_context_recall(question, ground_truth, gr_context)
    except Exception as e:
        print(f"GraphRAG Failed: {e}")
        gr_score = 0.0

    # 2. Prolog GraphRAG
    try:
        pgr_result = run_prolog_graphrag(question, flag="prolog")
        contexts = pgr_result.get("contexts", [])
        pgr_context = "\n".join(contexts) if isinstance(contexts, list) else str(contexts)
        pgr_score, pgr_reason = evaluate_context_recall(question, ground_truth, pgr_context)
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