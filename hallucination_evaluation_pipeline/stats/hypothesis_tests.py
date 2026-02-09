from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.contingency_tables import cochrans_q
from config import *
import numpy as np
import pandas as pd
import csv
import os

from stats.visualizer_generator import generateBoxPlotGrids

# Define top-level directory (one level up from current script)
TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define target folder inside top-level directory
IO_DIR = os.path.join(TOP_LEVEL_DIR, "experimental-data")

def computeHypothesisTestForContinuousEval(tuned_data: np.ndarray, graphrag_data: np.ndarray, prolog_graphrag_data: np.ndarray, eval: Metric | Benchmark):
    """
    Performs hypothesis testing (e.g., t-test or Mann-Whitney U test) for a given continuous evaluation criterion
    across different pipelines.
    """
    
    stat, p_value = kruskal(tuned_data, graphrag_data, prolog_graphrag_data)
    interpretation = "SIGNIFICANT" if p_value < ALPHA else "NON_SIGNIFICANT"
    
    # Save to CSV
    with open(os.path.join(IO_DIR, "hypothesis_test_results.csv"), 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["eval_name", "p_value", "interpretation"]
        )
        writer.writerow({
            "eval_name": eval.value.lower().replace(" ", "_"),
            "p_value": p_value,
            "interpretation": interpretation
        })
                
        
def performContinuousHypothesisTests():
    df_map = {
        Metric.FAITHFULNESS: pd.read_csv(r"../hallucination-evaluation-results/faithfulness_scores.csv"),
        Metric.ANSWER_RELEVANCE: pd.read_csv(r"../hallucination-evaluation-results/answer_relevance_scores.csv"),
        Metric.CONTEXT_RECALL: pd.read_csv(r"../hallucination-evaluation-results/context_recall_scores.csv"),
        Benchmark.TRUTHFULQA: pd.read_csv(r"../hallucination-evaluation-results/truthfulqa_scores.csv")
    }

    for metric in list(Metric):
        tuned_data = np.array(df_map[Pipeline.TUNED]["tuned_score"])
        graphrag_data = np.array(df_map[Pipeline.GRAPHRAG]["graphrag_score"])
        prolog_graphrag_data = np.array(df_map[Pipeline.PROLOG_GRAPHRAG]["prolog_graphrag_score"])

        computeHypothesisTestForContinuousEval(
            tuned_data=tuned_data,
            graphrag_data=graphrag_data,
            prolog_graphrag_data=prolog_graphrag_data,
            metric=metric,
        )
            
    generateBoxPlotGrids(
        tuned_data=tuned_data,
        graphrag_data=graphrag_data,
        prolog_graphrag_data=prolog_graphrag_data,
    )
    
    
def computeHypothesisTestForDiscreteEval(tuned_data: np.ndarray, graphrag_data: np.ndarray, prolog_graphrag_data: np.ndarray, eval: Metric):
    """
    Performs hypothesis testing (e.g., Cochran's Q test) for a given discrete evaluation criterion
    across different pipelines.
    """

    data = np.array(tuned_data.tolist(), graphrag_data.tolist(), prolog_graphrag_data.tolist())
    
    stat, p_value, df = cochrans_q(data)
    interpretation = "SIGNIFICANT" if p_value < ALPHA else "NON_SIGNIFICANT"
    
    # Save to CSV
    with open(os.path.join(IO_DIR, "hypothesis_test_results.csv"), 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["eval_name", "p_value", "interpretation"]
        )
        writer.writerow({
            "eval_name": eval.value.lower().replace(" ", "_"),
            "p_value": p_value,
            "interpretation": interpretation
        })
                
        
def performDiscreteHypothesisTests():
    df_map = {
        Benchmark.MMLU: pd.read_csv(r"../hallucination-evaluation-results/mmlu_scores.csv"),
        Benchmark.HOTPOTQA: pd.read_csv(r"../hallucination-evaluation-results/hotpotqa_scores.csv")
    }

    for metric in list(Metric):
        tuned_data = np.array(df_map[Pipeline.TUNED]["tuned_score"])
        graphrag_data = np.array(df_map[Pipeline.GRAPHRAG]["graphrag_score"])
        prolog_graphrag_data = np.array(df_map[Pipeline.PROLOG_GRAPHRAG]["prolog_graphrag_score"])

        computeHypothesisTestForDiscreteEval(
            tuned_data=tuned_data,
            graphrag_data=graphrag_data,
            prolog_graphrag_data=prolog_graphrag_data,
            metric=metric,
        )
            
    generateBoxPlotGrids(
        tuned_data=tuned_data,
        graphrag_data=graphrag_data,
        prolog_graphrag_data=prolog_graphrag_data,
    )

def performHypothesisTests():
    performContinuousHypothesisTests()
    performDiscreteHypothesisTests()