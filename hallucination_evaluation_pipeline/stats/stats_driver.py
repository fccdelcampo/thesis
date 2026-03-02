import os
import sys
import pandas as pd
import numpy as np
import random
from pathlib import Path

# Project root and stats directory setup
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
ROOT = Path(_current_dir).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from config import CLEAR_DATA_PASSWORD, Metric, Benchmark, CONTINUOUS_EVALS, DISCRETE_EVALS, NUM_TEST_CASES
    from normality_tests import perform_normality_tests
    from hypothesis_tests import perform_hypothesis_tests
    from posthoc_tests import perform_posthoc_tests
except ImportError:
    from .config import CLEAR_DATA_PASSWORD, Metric, Benchmark, CONTINUOUS_EVALS, DISCRETE_EVALS, NUM_TEST_CASES
    from .normality_tests import perform_normality_tests
    from .hypothesis_tests import perform_hypothesis_tests
    from .posthoc_tests import perform_posthoc_tests

# Define target folder inside top-level directory
DATA_DIR = ROOT / "hallucination_evaluation_results/data"

    

def ensure_csv_headers():
    """Ensure all stats CSV files have their proper headers before writing data."""
    headers = {
        "answer_relevancy_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "context_recall_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "faithfulness_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "ai2_arc_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "mmlu_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "openbookqa_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "normality_test_results.csv": "pipeline,eval_name,p_value,normality\n",
        "hypothesis_test_results.csv": "eval_name,p_value,interpretation\n",
        "posthoc_test_results.csv": "eval_name,loser,winner,p_value\n",
        
    }
    
    for filename, header in headers.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(header)
                
def clear_datasets():
    password = input("You're about to clear all datasets under explainability_evaluation_results. Enter data clearance password -> ")
    if password != CLEAR_DATA_PASSWORD:
        sys.exit("Incorrect password.")
    headers = {
        "answer_relevancy_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "context_recall_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "faithfulness_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "ai2_arc_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "mmlu_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "openbookqa_scores.csv": "test_case_id,tuned_score,tuned_label,graphrag_score,graphrag_label,prolog_graphrag_score,prolog_graphrag_label\n",
        "normality_test_results.csv": "pipeline,eval_name,p_value,normality\n",
        "hypothesis_test_results.csv": "eval_name,p_value,interpretation\n",
        "posthoc_test_results.csv": "eval_name,loser,winner,p_value\n",
        
    }
    
    for filename, header in headers.items():
        path = os.path.join(DATA_DIR, filename)
        with open(path, 'w', encoding='utf-8', newline='') as f:
            pass

def perform_statistical_tests():
    ensure_csv_headers()
    
    print("Starting Normality Tests...")
    perform_normality_tests()
    
    print("Starting Hypothesis Tests...")
    perform_hypothesis_tests()
    
    print("Starting Post Hoc Tests...")
    perform_posthoc_tests()
    
    print("Statistical Tests Completed.")
    
    
def generate_artificial_data():
    # For <pipeline>_results.csv files
    # workload,awt,art,throughput,cpu_utilization
    print("Generating artificial data...")
    for eval in list(Metric) + list(Benchmark):
        for i in range(NUM_TEST_CASES):
            with open(DATA_DIR / f"{eval.value[1]}_scores.csv", 'a', newline='') as f:
                # test_case_id,tuned_score,graphrag_score,prolog_graphrag_score
                if eval in CONTINUOUS_EVALS:
                    tuned_score, graphrag_score, prolog_graphrag_score = np.random.uniform(0, 0.33), np.random.uniform(0.34, 0.66), np.random.uniform(0.67, 1)
                    writer = pd.DataFrame({
                        "test_case_id": i,
                        "tuned_score": [tuned_score],
                        "tuned_label": [0 if tuned_score > 0.7 else 1],
                        "graphrag_score": [graphrag_score],
                        "graphrag_label": [0 if graphrag_score > 0.7 else 1],
                        "prolog_graphrag_score": [prolog_graphrag_score],
                        "prolog_graphrag_label": [0 if prolog_graphrag_score > 0.7 else 1],
                    })
                else:
                    tuned_score, graphrag_score, prolog_graphrag_score = 0, random.randint(0, 1), 1
                    writer = pd.DataFrame({
                        "test_case_id": i,
                        "tuned_score": [tuned_score],
                        "tuned_label": [0 if tuned_score == 1 else 1],
                        "graphrag_score": [graphrag_score],
                        "graphrag_label": [0 if graphrag_score == 1 else 1],
                        "prolog_graphrag_score": [prolog_graphrag_score],
                        "prolog_graphrag_label": [0 if prolog_graphrag_score == 1 else 1]
                    })
                writer.to_csv(f, header=f.tell()==0, index=False)
    print("Done generating artificial data.")
    
def informal_tests():
    print("================Starting informal tests...================")
    clear_datasets()
    ensure_csv_headers()
    generate_artificial_data()
    perform_statistical_tests()
  
if __name__ == "__main__":  
    informal_tests()