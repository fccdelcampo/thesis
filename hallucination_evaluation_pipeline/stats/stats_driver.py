import os
import sys
import pandas as pd
import numpy as np
import random

# Define top-level directory (one level up from current script)
TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(TOP_LEVEL_DIR)
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Append top-level directory to sys.path BEFORE importing
# sys.path.append(TOP_LEVEL_DIR)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# RESULTS_DIR = ROOT / "hallucination_evaluation_results"
sys.path.append(str(ROOT))
# Define target folder inside top-level directory
IO_DIR = ROOT / "hallucination_evaluation_results"

try:
    from .config import *
    from .normality_tests import perform_normality_tests
    from .hypothesis_tests import perform_hypothesis_tests
    from .posthoc_tests import perform_posthoc_tests
except ImportError as e:
    print(f"Relative import failed: {e}")
    try:
        from config import *
        from normality_tests import perform_normality_tests
        from hypothesis_tests import perform_hypothesis_tests
        from posthoc_tests import perform_posthoc_tests
    except ImportError as e2:
        print(f"Absolute import failed: {e2}")
        from stats.config import *
        from stats.normality_tests import perform_normality_tests
        from stats.hypothesis_tests import perform_hypothesis_tests
        from stats.posthoc_tests import perform_posthoc_tests

    

def ensure_csv_headers():
    """Ensure all stats CSV files have their proper headers before writing data."""
    headers = {
        "answer_relevance_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "context_recall_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "faithfulness_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "hotpotqa_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "mmlu_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "truthfulqa_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "normality_test_results.csv": "pipeline,eval_name,p_value,normality\n",
        "hypothesis_test_results.csv": "eval_name,p_value,interpretation\n",
        "posthoc_test_results.csv": "eval_name,loser,winner,p_value\n",
        
    }
    
    for filename, header in headers.items():
        path = os.path.join(IO_DIR, filename)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(header)
                
def clear_datasets():
    headers = {
        "answer_relevance_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "context_recall_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "faithfulness_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "hotpotqa_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "mmlu_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "truthfulqa_scores.csv": "test_case_id,tuned_score,graphrag_score,prolog_graphrag_score\n",
        "normality_test_results.csv": "pipeline,eval_name,p_value,normality\n",
        "hypothesis_test_results.csv": "eval_name,p_value,interpretation\n",
        "posthoc_test_results.csv": "eval_name,loser,winner,p_value\n",
        
    }
    
    for filename, header in headers.items():
        path = os.path.join(IO_DIR, filename)
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
            with open(rf"hallucination-evaluation-results/{eval.value[1]}_scores.csv", 'a', newline='') as f:
                # test_case_id,tuned_score,graphrag_score,prolog_graphrag_score
                if eval in CONTINUOUS_EVALS:
                    writer = pd.DataFrame({
                        "test_case_id": i,
                        "tuned_score": [np.random.uniform(6, 7)],
                        "graphrag_score": [np.random.uniform(0.7, 1.0)],
                        "prolog_graphrag_score": [np.random.uniform(50, 150)]
                    })
                else:
                    writer = pd.DataFrame({
                        "test_case_id": i,
                        "tuned_score": [random.randint(0, 1)],
                        "graphrag_score": [random.randint(0, 1)],
                        "prolog_graphrag_score": [random.randint(0, 1)]
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