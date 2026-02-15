import json
import os
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def get_embedding(text: str):
    """
    Get embedding for a text using Ollama (nomic-embed-text or similar).
    """
    try:
        import ollama
        # Use a model that supports embeddings. 
        # Verify if 'llama3' supports it or use 'nomic-embed-text' if available.
        # Fallback to 'llama3:8b-instruct-fp16' if specific embed model not set,
        # but llama3 is instructed model.
        # Ideally use mxbai-embed-large or nomic-embed-text.
        # For this env, let's try 'nomic-embed-text' or 'all-minilm'.
        # If not, we might need to skip or use a simple overlap metric.
        # Let's try to use the 'graphrag_pipeline.retriever.embedder' if possible.
        
        # Simple fix: Use ollama.embeddings
        resp = ollama.embeddings(model='llama3:8b-instruct-fp16', prompt=text)
        return resp['embedding']
    except Exception as e:
        # print(f"Embedding failed: {e}")
        return [0.0] * 4096 # Dummy, size depends on model

def evaluate_answer_relevance(question: str, answer: str) -> tuple[float, str]:
    """
    Evaluates Answer Relevance by generating artificial questions from the answer
    and computing cosine similarity with the original question.
    """
    if not answer or len(answer) < 5:
        return 0.0, "Answer too short"

    prompt = f"""
    Generate a question for which the following text is a suitable answer.
    
    ANSWER: {answer}
    
    OUTPUT: Return ONLY the generated question.
    """
    
    try:
        import ollama
        response = ollama.chat(
            model='llama3:8b-instruct-fp16',
            messages=[{'role': 'system', 'content': 'You are a question generator.'}, 
                      {'role': 'user', 'content': prompt}],
            options={'temperature': 0.7}
        )
        generated_question = response['message']['content'].strip()
        
        # Calculate Cosine Similarity between Original Question and Generated Question
        q_emb = get_embedding(question)
        gq_emb = get_embedding(generated_question)
        
        if len(q_emb) == len(gq_emb) and len(q_emb) > 1:
            similarity = cosine_similarity([q_emb], [gq_emb])[0][0]
            # Normalize to 0-1 if needed, usually cosine is -1 to 1 but for embeddings usually 0-1
            score = max(0.0, similarity)
            return float(score), generated_question
        else:
            return 0.0, "Embedding Mismatch"
            
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return 0.0, "Error"

def run_test_case(test_case: dict, test_case_id: int):
    question = test_case.get("question")
    if not question:
        return

    print(f"Running Answer Relevance {test_case_id}: {question[:50]}...")

    # 1. GraphRAG
    try:
        gr_result = run_graphrag(question)
        gr_answer = gr_result.get("answer", "")
        gr_score, gr_gen_q = evaluate_answer_relevance(question, gr_answer)
    except Exception as e:
        print(f"GraphRAG Failed: {e}")
        gr_score = 0.0

    # 2. Prolog GraphRAG
    try:
        pgr_result = run_prolog_graphrag(question, flag="prolog")
        pgr_answer = pgr_result.get("answer", "")
        pgr_score, pgr_gen_q = evaluate_answer_relevance(question, pgr_answer)
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