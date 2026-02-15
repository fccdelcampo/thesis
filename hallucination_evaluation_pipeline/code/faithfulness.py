import json
import os
import sys
import pandas as pd
from tqdm import tqdm
import time

# Add project root to sys.path to allow importing from parallel directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prolog_graphrag_pipeline.llm import generate as llm_generate
from prolog_graphrag_pipeline.config import LLM_SYSTEM_PROMPT

# Define paths
# test_cases/ is two levels up from this file
TEST_CASES_DIR = os.path.join(project_root, "test_cases")
RESULTS_DIR = os.path.join(project_root, "hallucination_evaluation_results")

def load_test_cases() -> list[dict]:
    """
    Reads .jsonl files in test_cases/ and turns each JSON line into a dict.
    Specifically targets faithfulness.jsonl (or creates synthetic ones if empty).
    For now, we will use mmlu.jsonl as a proxy if faithfulness.jsonl is empty, 
    or just return an empty list if strict. The plan says use faithfulness.jsonl.
    """
    file_path = os.path.join(TEST_CASES_DIR, "faithfulness.jsonl")
    test_cases = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_cases.append(json.loads(line))
    return test_cases

def evaluate_faithfulness(question: str, answer: str, context: str) -> tuple[float, str]:
    """
    Uses LLM-as-a-judge to evaluate faithfulness.
    Returns a score (0.0 or 1.0) and a reason/label.
    """
    if not context:
        return 0.0, "No context provided"

    prompt = f"""
    You are a judge evaluating the faithfulness of an answer to a given context.
    
    QUESTION: {question}
    CONTEXT: {context}
    ANSWER: {answer}
    
    INSTRUCTION:
    1. Extract all claims made in the ANSWER.
    2. For each claim, check if it is supported by the CONTEXT.
    3. If ALL claims are supported, the answer is FAITHFUL.
    4. If ANY claim is contradicting or not found in the context, the answer is HALLUCINATED.
    
    OUTPUT FORMAT:
    Return ONLY a JSON object with keys "score" (0 or 1) and "reason".
    Example: {{"score": 1, "reason": "All claims supported."}}
    """
    
    try:
        # We use a simple generation call here. 
        # Ideally, we should use a separate 'judge' model or the same LLM with high temp.
        # For simplicity, reusing the existing llm interface but we need to mock 'retrieved_context' arg
        # or use the llm.generate raw logic. 
        # Since llm.generate is tied to the RAG pipeline, we might need a direct call.
        # But let's try to use the 'q' flag or similar if available, or just direct import.
        import ollama
        response = ollama.chat(
            model='llama3:8b-instruct-fp16',
            messages=[{'role': 'system', 'content': 'You are a strict evaluator.'}, 
                      {'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )
        content = response['message']['content']
        # Try to parse JSON
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = content[start:end]
            data = json.loads(json_str)
            return float(data.get("score", 0)), data.get("reason", "Parsed")
        else:
            return 0.0, "Failed to parse JSON"
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0.0, "Error"

# GraphRAG
from graphrag_pipeline.graphrag.graphrag_driver import run_pipeline as run_graphrag

# Prolog GraphRAG
# Import as package from project root
from prolog_graphrag_pipeline.main_driver import run_pipeline as run_prolog_graphrag

# Tuned Pipeline (Assuming it exists or we skip)
# For now, we'll skip tuned or mock it as 0 to avoid ImportErrors if not ready.
# If you have a tuned pipeline, import it here.

def run_test_case(test_case: dict, test_case_id: int):
    """
    Runs a test case by calling each pipeline's run_pipeline function.
    """
    question = test_case.get("question")
    if not question:
        return

    print(f"Running Test Case {test_case_id}: {question[:50]}...")

    # 1. Run GraphRAG Pipeline
    try:
        # Expected return: dict with "answer", "context"
        graphrag_result = run_graphrag(question) 
        graphrag_answer = graphrag_result.get("answer", "")
        graphrag_context = graphrag_result.get("context", "") # Assuming structure
        
        # Evaluate Faithfulness
        gr_score, gr_reason = evaluate_faithfulness(question, graphrag_answer, graphrag_context)
    except Exception as e:
        print(f"GraphRAG Failed: {e}")
        gr_score = 0.0
    
    # 2. Run Prolog GraphRAG Pipeline
    try:
        # Expected return: dict with "answer", "contexts" (list)
        # run_pipeline(user_prompt: str, flag: str, sample_mode: bool = False, include_conclusion: bool = True)
        prolog_result = run_prolog_graphrag(question, flag="prolog") # using 'prolog' flag as default
        prolog_answer = prolog_result.get("answer", "")
        # content is likely a list or string
        contexts = prolog_result.get("contexts", [])
        prolog_context = "\n".join(contexts) if isinstance(contexts, list) else str(contexts)
        
        # Evaluate Faithfulness
        pgr_score, pgr_reason = evaluate_faithfulness(question, prolog_answer, prolog_context)
    except Exception as e:
        print(f"Prolog GraphRAG Failed: {e}")
        pgr_score = 0.0

    # 3. Tuned Pipeline (Placeholder)
    tuned_score = 0.0 
    
    # Log Results
    log_results(test_case_id, tuned_score, gr_score, pgr_score)
    
    # Log Q&A (Optional/TODO: Implement checking against ground truth or just logging)
    # log_question_and_answer(...) 
    
    pass 

def log_results(test_case_id: int, tuned_score: float, graphrag_score: float, prolog_graphrag_score: float) -> None:
    file_path = os.path.join(RESULTS_DIR, "faithfulness_scores.csv")
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
    print(f"Loaded {len(test_cases)} test cases for Faithfulness.")
    
    # Since we can't easily run all pipelines without circular imports or heavy setup in this snip,
    # We will iterate and print for now, or assume we are called by main_driver which handles imports?
    # No, main_driver calls this.main().
    # So this script is responsible for driving the test.
    
    for i, test_case in enumerate(tqdm(test_cases)):
        run_test_case(test_case, i)
        # pass

if __name__ == "__main__":
    main()