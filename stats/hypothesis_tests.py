from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
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

def computeHypothesisTestForContinuousEval(fcfs_ndarray, rr_ndarray, osjfdrr_ndarray, srr_ndarray, kdts_ndarray, metric: Metric, workload: Workload):
    """
    Performs hypothesis testing (e.g., t-test or Mann-Whitney U test) for a given continuous evaluation criterion
    across different pipelines.

    Args:
        fcfs_ndarray (np.ndarray): Data for FCFS pipeline.
        rr_ndarray (np.ndarray): Data for RR pipeline.
        osjfdrr_ndarray (np.ndarray): Data for oSJFDRR pipeline.
        kdts_ndarray (np.ndarray): Data for KDTS pipeline.
        metric (Metric): The performance metric being tested.
    """
    other_pipelines = []
    print("Current metric: ", metric.value)
    match(metric):
        case Metric.AWT:
            other_pipelines = [(Pipeline.FCFS, fcfs_ndarray), (Pipeline.RR, rr_ndarray), (Pipeline.oSJFDRR, osjfdrr_ndarray), (Pipeline.SRR, srr_ndarray)]
        case Metric.ART:
            other_pipelines = [(Pipeline.FCFS, fcfs_ndarray)]
        case Metric.CPU_UTILIZATION:
            other_pipelines = [(Pipeline.RR, rr_ndarray), (Pipeline.oSJFDRR, osjfdrr_ndarray), (Pipeline.SRR, srr_ndarray)]
        case default:
            raise ValueError(f"Unsupported metric: {metric}")
    print(f"METRIC: {metric.value}, WORKLOAD: {workload.value}")
        
    # workload,metric,other_pipeline,p_value,interpretation
    comparisons = []   # list of dicts

    for other in other_pipelines:
        # Expected-direction test
        if metric != Metric.CPU_UTILIZATION:          # lower is better (AWT)
            p_exp = mannwhitneyu(kdts_ndarray, other[1], alternative='less').pvalue
            p_opp = mannwhitneyu(kdts_ndarray, other[1], alternative='greater').pvalue
        else:                                         # higher is better (CPU)
            p_exp = mannwhitneyu(kdts_ndarray, other[1], alternative='greater').pvalue
            p_opp = mannwhitneyu(kdts_ndarray, other[1], alternative='less').pvalue

        comparisons.append({
            "other": other[0].value,
            "p_exp": p_exp,
            "p_opp": p_opp
        })

    # Flatten all p-values for Holm correction
    raw_pvals = [c["p_exp"] for c in comparisons] + \
                [c["p_opp"] for c in comparisons]

    # Holm correction
    _, corrected, _, _ = multipletests(raw_pvals, method="holm")

    # Split corrected p-values back
    corrected_exp = corrected[:len(comparisons)]
    corrected_opp = corrected[len(comparisons):]

    # Assign interpretations AFTER correction
    for i, c in enumerate(comparisons):
        p_exp_corr = corrected_exp[i]
        p_opp_corr = corrected_opp[i]

        if p_exp_corr < ALPHA and p_opp_corr < ALPHA:
            raise ValueError("Both directions significant after correction, check data.")
        
        if p_exp_corr < ALPHA:
            interpretation = "win"
            final_p = p_exp_corr
        elif p_opp_corr < ALPHA:
            interpretation = "loss"
            final_p = p_opp_corr
        else:
            interpretation = "not_significant"
            final_p = min(p_exp_corr, p_opp_corr)

        # Save to CSV
        with open(os.path.join(IO_DIR, "hypothesis_test_results.csv"), 'a', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["workload", "metric", "other_pipeline", "p_value", "interpretation"]
            )
            writer.writerow({
                "workload": workload.value,
                "metric": metric.value.lower().replace(" ", "_"),
                "other_pipeline": c["other"],
                "p_value": final_p,
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
        prolog_graphrag_data = np.array(df_map[Pipeline.PROLOG_GRAPHRAG]["prolog_graphrag_score"]])

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
    