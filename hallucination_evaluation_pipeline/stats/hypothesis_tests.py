from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.contingency_tables import cochrans_q
from config import *
import numpy as np
import pandas as pd
import csv
import os
from visualizer_generator import generate_box_plot_grids, generate_bar_graph_grid

# from stats.visualizer_generator import generateBoxPlotGrids

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# RESULTS_DIR = ROOT / "hallucination_evaluation_results"
sys.path.append(str(ROOT))
# Define target folder inside top-level directory
DATA_DIR = ROOT / "hallucination_evaluation_results/data"

def compute_hypothesis_test_for_continuous_eval(tuned_data: np.ndarray, graphrag_data: np.ndarray, prolog_graphrag_data: np.ndarray, eval: Metric | Benchmark):
    """
    Performs hypothesis testing (e.g., t-test or Mann-Whitney U test) for a given continuous evaluation criterion
    across different pipelines.
    """
    
    stat, p_value = kruskal(tuned_data, graphrag_data, prolog_graphrag_data)
    interpretation = "SIGNIFICANT" if p_value < ALPHA else "NON_SIGNIFICANT"
    
    # Save to CSV
    with open(os.path.join(DATA_DIR, "hypothesis_test_results.csv"), 'a', newline='') as f:
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
        Metric.FAITHFULNESS: pd.read_csv(DATA_DIR / "faithfulness_scores.csv"),
        Metric.ANSWER_RELEVANCE: pd.read_csv(DATA_DIR / "answer_relevance_scores.csv"),
        Metric.CONTEXT_RECALL: pd.read_csv(DATA_DIR / "context_recall_scores.csv"),
        Benchmark.TRUTHFULQA: pd.read_csv(DATA_DIR / "truthfulqa_scores.csv")
    }

    for eval in CONTINUOUS_EVALS:
        tuned_data = np.array(df_map[eval]["tuned_score"])
        graphrag_data = np.array(df_map[eval]["graphrag_score"])
        prolog_graphrag_data = np.array(df_map[eval]["prolog_graphrag_score"])

        compute_hypothesis_test_for_continuous_eval(
            tuned_data=tuned_data,
            graphrag_data=graphrag_data,
            prolog_graphrag_data=prolog_graphrag_data,
            eval=eval,
        )
            
    generate_box_plot_grids()
    
    
def compute_hypothesis_test_for_discrete_eval(tuned_data: np.ndarray, graphrag_data: np.ndarray, prolog_graphrag_data: np.ndarray, eval: Metric):
    """
    Performs hypothesis testing (e.g., Cochran's Q test) for a given discrete evaluation criterion
    across different pipelines.
    """

    data = np.array([tuned_data.tolist(), graphrag_data.tolist(), prolog_graphrag_data.tolist()])
    
    result = cochrans_q(data)
    p_value = result.pvalue
    interpretation = "SIGNIFICANT" if p_value < ALPHA else "NON_SIGNIFICANT"
    
    # Save to CSV
    with open(os.path.join(DATA_DIR, "hypothesis_test_results.csv"), 'a', newline='') as f:
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
        Benchmark.MMLU: pd.read_csv(DATA_DIR / "mmlu_scores.csv"),
        Benchmark.HOTPOTQA: pd.read_csv(DATA_DIR / "hotpotqa_scores.csv")
    }

    for eval in DISCRETE_EVALS:
        tuned_data = np.array(df_map[eval]["tuned_score"])
        graphrag_data = np.array(df_map[eval]["graphrag_score"])
        prolog_graphrag_data = np.array(df_map[eval]["prolog_graphrag_score"])

        compute_hypothesis_test_for_discrete_eval(
            tuned_data=tuned_data,
            graphrag_data=graphrag_data,
            prolog_graphrag_data=prolog_graphrag_data,
            eval=eval,
        )
            
    generate_bar_graph_grid()

def perform_hypothesis_tests():
    perform_continuous_hypothesis_tests()
    perform_discrete_hypothesis_tests()