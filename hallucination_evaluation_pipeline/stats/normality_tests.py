import os
import sys
import numpy as np
import csv
import pandas as pd
from pathlib import Path

# Scipy imports FIRST (before wildcard import)
from scipy.stats import gmean, shapiro

# Define top-level directory (one level up from current script)
TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append top-level directory to sys.path BEFORE importing
sys.path.append(TOP_LEVEL_DIR)
ROOT = Path(__file__).resolve().parents[2]
# RESULTS_DIR = ROOT / "hallucination_evaluation_results"
sys.path.append(str(ROOT))

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
# Define target folder inside top-level directory
IO_DIR = ROOT / "hallucination_evaluation_results"

def shapiro_test(data: np.ndarray, pipeline: Pipeline, eval: Metric):
    if len(data) < 3:
        return np.nan, "TOO_SMALL_N"
    _, p_value = shapiro(data)
    normality = "NORMAL" if p_value > ALPHA else "NOT_NORMAL"
        
    # Write normality test results to CSV
    normality_row = {
        "pipeline": pipeline.value[1],
        "eval_name": eval.value[1],
        "p_value": p_value,
        "normality": normality
    }

    with open(os.path.join(IO_DIR, "normality_test_results.csv"), 'a') as f:
        writer = csv.DictWriter(f, fieldnames=normality_row.keys())
        writer.writerow(normality_row)

from typing import Union

def compute_normality_test(df: pd.DataFrame, eval: Union[Metric, Benchmark]):
    for pipeline in list(Pipeline):
        col_name = pipeline.value[1] + "_score"
        data = df[col_name].to_numpy()
        shapiro_test(data, pipeline, eval)

def perform_normality_tests():
    df_map = {
        Metric.FAITHFULNESS: pd.read_csv(IO_DIR / "faithfulness_scores.csv"),
        Metric.ANSWER_RELEVANCE: pd.read_csv(IO_DIR / "answer_relevance_scores.csv"),
        Metric.CONTEXT_RECALL: pd.read_csv(IO_DIR / "context_recall_scores.csv"),
        Benchmark.TRUTHFULQA: pd.read_csv(IO_DIR / "truthfulqa_scores.csv")
    }

    for eval, df in df_map.items():
        compute_normality_test(df, eval)
    visualizer_generator.generate_qq_grids()
    