import os
import sys
import numpy as np
import csv
import pandas as pd

# Scipy imports FIRST (before wildcard import)
from scipy.stats import gmean, shapiro

# Define top-level directory (one level up from current script)
TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append top-level directory to sys.path BEFORE importing
sys.path.append(TOP_LEVEL_DIR)

# Import classes and config (but this brings in stats from standard library)
from config import *
from .visualizer_generator import generate_qq_grid_for_eval
import pandas
# Define target folder inside top-level directory
IO_DIR = os.path.join(TOP_LEVEL_DIR, "hallucination-evaluation-results")

def shapiro_test(data: np.ndarray, pipeline: Pipeline, eval: Metric):
    if len(data) < 3:
        return np.nan, "TOO_SMALL_N"
    _, p_value = shapiro(data)
    normality = "NORMAL" if p_value > ALPHA else "NOT_NORMAL"
        
    # Write normality test results to CSV
    normality_row = {
        "pipeline": pipeline.value.replace(" Pipeline", "").lower().replace("-", "_").replace(" ", "_"),
        "eval_name": eval.value.lower().replace(" ", "_"),
        "p_value": p_value,
        "normality": normality
    }

    with open(os.path.join(IO_DIR, "normality_test_results.csv"), 'a') as f:
        writer = csv.DictWriter(f, fieldnames=normality_row.keys())
        writer.writerow(normality_row)

def computeNormalityTest(df: pandas.DataFrame, eval: Metric | Benchmark):
    for pipeline in list(Pipeline):
        col_name = pipeline.value.lower().replace("-", "_").replace(" ", "_").replace("pipeline", "") + "score"
        data = df[col_name].to_numpy()
        shapiro_test(data, pipeline, eval)

def performNormalityTests():
    df_map = {
        Metric.FAITHFULNESS: pd.read_csv(r"../../hallucination-evaluation-results/faithfulness_scores.csv"),
        Metric.ANSWER_RELEVANCE: pd.read_csv(r"../../hallucination-evaluation-results/answer_relevance_scores.csv"),
        Metric.CONTEXT_RECALL: pd.read_csv(r"../../hallucination-evaluation-results/context_recall_scores.csv"),
        Benchmark.TRUTHFULQA: pd.read_csv(r"../../hallucination-evaluation-results/truthfulqa_scores.csv")
    }

    for eval, df in df_map.items():
        computeNormalityTest(df, eval)

    for eval in CONTINUOUS_EVALS:
        generate_qq_grid_for_eval(
            eval=eval,
            df_map=df_map
        )
    