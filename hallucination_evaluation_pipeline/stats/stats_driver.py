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
# RESULTS_DIR = ROOT / "hallucination-evaluation-results"
sys.path.append(str(ROOT))
# Define target folder inside top-level directory
IO_DIR = ROOT / "hallucination-evaluation-results"

from config import *
from normality_tests import performNormalityTests
from hypothesis_tests import performHypothesisTests
from posthoc_tests import performPostHocTests

    

def ensureCsvHeaders():
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

def performStatisticalTests():
    
    print("Starting Normality Tests...")
    performNormalityTests()
    
    print("Starting Hypothesis Tests...")
    performHypothesisTests()
    
    print("Starting Post Hoc Tests...")
    performPostHocTests()
    
    print("Statistical Tests Completed.")
    
def generateArtificialData():
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
    ensureCsvHeaders()
    generateArtificialData()
    performStatisticalTests()
  
if __name__ == "__main__":  
    informal_tests()