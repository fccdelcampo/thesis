import sys
import os
import numpy as np
import pandas as pd
import csv
STATS_AVAILABLE = True
try:
    from scipy.stats import kruskal, mannwhitneyu
    from statsmodels.stats.contingency_tables import cochrans_q
except ImportError:
    STATS_AVAILABLE = False
    print("Warning: scipy or statsmodels not found. Hypothesis tests will be skipped.")
try:
    from .config import *
    from . import visualizer_generator
except ImportError:
    try:
        from config import *
        import visualizer_generator
    except ImportError:
        from stats.config import *
        from stats import visualizer_generator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# RESULTS_DIR = ROOT / "hallucination_evaluation_results"
sys.path.append(str(ROOT))
# Define target folder inside top-level directory
IO_DIR = ROOT / "hallucination_evaluation_results"

from typing import Union

def compute_hypothesis_test_for_continuous_eval(tuned_data: np.ndarray, graphrag_data: np.ndarray, prolog_graphrag_data: np.ndarray, eval: Union[Metric, Benchmark]):
    """
    Performs hypothesis testing (e.g., t-test or Mann-Whitney U test) for a given continuous evaluation criterion
    across different pipelines.
    """
    if not STATS_AVAILABLE:
        return
    
    try:
        stat, p_value = kruskal(tuned_data, graphrag_data, prolog_graphrag_data)
        interpretation = "SIGNIFICANT" if p_value < ALPHA else "NON_SIGNIFICANT"
    except Exception as e:
        print(f"Kruskal-Wallis test failed: {e}")
        p_value = np.nan
        interpretation = "ERROR"
    
    # Save to CSV
    with open(os.path.join(IO_DIR, "hypothesis_test_results.csv"), 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["eval_name", "p_value", "interpretation"]
        )
        writer.writerow({
            "eval_name": eval.value[1],
            "p_value": p_value,
            "interpretation": interpretation
        })
                
        
def perform_continuous_hypothesis_tests():
    df_map = {
        Metric.FAITHFULNESS: pd.read_csv(IO_DIR / "faithfulness_scores.csv"),
        Metric.ANSWER_RELEVANCE: pd.read_csv(IO_DIR / "answer_relevance_scores.csv"),
        Metric.CONTEXT_RECALL: pd.read_csv(IO_DIR / "context_recall_scores.csv"),
        Benchmark.TRUTHFULQA: pd.read_csv(IO_DIR / "truthfulqa_scores.csv")
    }

    for eval in CONTINUOUS_EVALS:
        tuned_data = visualizer_generator.clean_data_column(df_map[eval], "tuned_score")
        graphrag_data = visualizer_generator.clean_data_column(df_map[eval], "graphrag_score")
        prolog_graphrag_data = visualizer_generator.clean_data_column(df_map[eval], "prolog_graphrag_score")

        compute_hypothesis_test_for_continuous_eval(
            tuned_data=tuned_data,
            graphrag_data=graphrag_data,
            prolog_graphrag_data=prolog_graphrag_data,
            eval=eval,
        )
            
    visualizer_generator.generate_box_plot_grids()
    
    
def compute_hypothesis_test_for_discrete_eval(tuned_data: np.ndarray, graphrag_data: np.ndarray, prolog_graphrag_data: np.ndarray, eval: Metric):
    """
    Performs hypothesis testing (e.g., Cochran's Q test) for a given discrete evaluation criterion
    across different pipelines.
    """
    if not STATS_AVAILABLE:
        return

    data = np.array([tuned_data.tolist(), graphrag_data.tolist(), prolog_graphrag_data.tolist()])
    
    try:
        result = cochrans_q(data)
        p_value = result.pvalue
        interpretation = "SIGNIFICANT" if p_value < ALPHA else "NON_SIGNIFICANT"
    except Exception as e:
        print(f"Cochran's Q test failed: {e}")
        p_value = np.nan
        interpretation = "ERROR"
    
    # Save to CSV
    with open(os.path.join(IO_DIR, "hypothesis_test_results.csv"), 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["eval_name", "p_value", "interpretation"]
        )
        writer.writerow({
            "eval_name": eval.value[1],
            "p_value": p_value,
            "interpretation": interpretation
        })
                
        
def perform_discrete_hypothesis_tests():
    df_map = {
        Benchmark.MMLU: pd.read_csv(IO_DIR / "mmlu_scores.csv"),
        Benchmark.HOTPOTQA: pd.read_csv(IO_DIR / "hotpotqa_scores.csv")
    }

    for eval in DISCRETE_EVALS:
        tuned_data = visualizer_generator.clean_data_column(df_map[eval], "tuned_score")
        graphrag_data = visualizer_generator.clean_data_column(df_map[eval], "graphrag_score")
        prolog_graphrag_data = visualizer_generator.clean_data_column(df_map[eval], "prolog_graphrag_score")

        compute_hypothesis_test_for_discrete_eval(
            tuned_data=tuned_data,
            graphrag_data=graphrag_data,
            prolog_graphrag_data=prolog_graphrag_data,
            eval=eval,
        )
            
    # generateBoxPlotGrids(
    #     tuned_data=tuned_data,
    #     graphrag_data=graphrag_data,
    #     prolog_graphrag_data=prolog_graphrag_data,
    # )

def perform_hypothesis_tests():
    perform_continuous_hypothesis_tests()
    perform_discrete_hypothesis_tests()