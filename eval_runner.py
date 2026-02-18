import csv
import os
import time

# Metrics (Importing but relying on mock/existing system context)
try:
    from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall
except ImportError:
    try:
        from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall
    except ImportError as e:
        print(f"Warning: ragas not installed or import failed: {e}. Evaluation logic will simulate calls or fail.")

# Loaders
from loaders.mmlu_loader import MMLULoader
from loaders.truthfulqa_loader import TruthfulQALoader
from loaders.hotpotqa_loader import HotpotQALoader

import sys

# Ensure the pipeline modules can be imported
pipeline_dir = os.path.join(os.path.dirname(__file__), 'prolog_graphrag_pipeline')
sys.path.append(pipeline_dir)

# Import the REAL pipeline driver (GraphRAG Only)
try:
    from graphrag_pipeline import graphrag_driver
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import graphrag_driver: {e}")
    sys.exit(1)

class GraphRAGSystem:
    def query(self, input_text):
        try:
            # Run the actual pipeline (GraphRAG Only)
            return graphrag_driver.run_pipeline(input_text)
        except Exception as e:
            print(f"Error querying pipeline: {e}")
            return {
                "answer": "Error",
                "contexts": [],
                "error": str(e)
            }
            
def load_cases():
    data = MMLULoader().load_data()
    print(data)

def run_evaluation():
    # 1. Setup
    graph_rag = GraphRAGSystem()
    loaders = [
        MMLULoader(),
        TruthfulQALoader(),
        HotpotQALoader()
    ]
    
    results_dir = "test_cases/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "evaluation_log.csv")
    
    # CSV Header
    fieldnames = [
        "dataset", "question", "ground_truth", "model_answer", 
        "faithfulness_score", "answer_relevancy_score", 
        "context_recall_score", "mmlu_accuracy", "execution_time"
    ]

    print("Starting Evaluation Pipeline...")
    
    with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for loader in loaders:
            dataset_name = loader.__class__.__name__.replace("Loader", "")
            print(f"\nLoading {dataset_name}...")
            test_cases = loader.load_data()
            
            for i, case in enumerate(test_cases[:1]):
                print(f"  Processing {i+1}/{len(test_cases)}: {case['question']}...")
                
                # 2. Query System
                start_time = time.time()
                response = graph_rag.query(case['formatted_input'])
                duration = time.time() - start_time
                
                answer = response.get('answer', '')
                contexts = response.get('contexts', [])
                
                # 3. Smart Scoring Logic
                scores = {
                    "faithfulness_score": 0.0, # Default
                    "answer_relevancy_score": "N/A",
                    "context_recall_score": "N/A",
                    "mmlu_accuracy": "N/A"
                }

                # Metric: Faithfulness (Always run if we have contexts)
                if contexts:
                    # Ragas call placeholder
                    # score = faithfulness.score(answer, contexts)
                    scores["faithfulness_score"] = 0.5 # Mock
                
                # Metric: Answer Relevancy (Only TruthfulQA and HotpotQA)
                if case['dataset'] in ['truthfulqa', 'hotpotqa']:
                     # Ragas call placeholder
                     # score = answer_relevancy.score(case['question'], answer)
                     scores["answer_relevancy_score"] = 0.5 # Mock

                # Metric: Context Recall (Only if GT Context exists)
                if case['ground_truth_context'] and len(case['ground_truth_context']) > 0:
                    # Ragas call placeholder
                    # score = context_recall.score(case['ground_truth_context'], contexts)
                    scores["context_recall_score"] = 0.5 # Mock

                # Metric: MMLU Accuracy (Strick check)
                if case['dataset'] == 'mmlu':
                    # Check if first character matches ground truth (assumes GT is 'A', 'B', etc.)
                    # Logic: Normalize headers like "Answer: A" -> "A"
                    clean_answer = answer.strip().upper()
                    if clean_answer.startswith(case['ground_truth']):
                         scores["mmlu_accuracy"] = 1.0
                    else:
                         scores["mmlu_accuracy"] = 0.0

                # 4. Log Result
                row = {
                    "dataset": case['dataset'],
                    "question": case['question'],
                    "ground_truth": case['ground_truth'],
                    "model_answer": answer,
                    "faithfulness_score": scores['faithfulness_score'],
                    "answer_relevancy_score": scores['answer_relevancy_score'],
                    "context_recall_score": scores['context_recall_score'],
                    "mmlu_accuracy": scores['mmlu_accuracy'],
                    "execution_time": duration
                }
                writer.writerow(row)

    print(f"\nEvaluation complete. Results saved to {results_file}")

if __name__ == "__main__":
    # run_evaluation()
    load_cases()
