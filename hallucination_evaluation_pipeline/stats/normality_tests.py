import os
import sys
import numpy as np
import csv
import pandas as pd
from pathlib import Path

# Project root and stats directory setup
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
ROOT = Path(_current_dir).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from config import ALPHA, Metric, Benchmark, Pipeline
    import visualizer_generator
    from visualizer_generator import generate_qq_grids
except ImportError:
    from .config import ALPHA, Metric, Benchmark, Pipeline
    from . import visualizer_generator
    from .visualizer_generator import generate_qq_grids

# Scipy imports FIRST (before wildcard import)
from scipy.stats import gmean, shapiro

# Define target folder inside top-level directory
DATA_DIR = ROOT / "hallucination_evaluation_results/data"

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

    with open(os.path.join(DATA_DIR, "normality_test_results.csv"), 'a') as f:
        writer = csv.DictWriter(f, fieldnames=normality_row.keys())
        writer.writerow(normality_row)

from typing import Union

def compute_normality_test(df: pd.DataFrame, eval: Union[Metric, Benchmark]):
    for pipeline in list(Pipeline):
        col_name = pipeline.value[1] + "_score"
        
        # Robust parsing for mixed types (floats, strings, stringified lists)
        raw_data = df[col_name].dropna().tolist()
        cleaned_data = []
        
        import ast
        for x in raw_data:
            try:
                if isinstance(x, (int, float)):
                    cleaned_data.append(float(x))
                elif isinstance(x, str):
                    # Try to parse stringified list like "[0.83]"
                    x = x.strip()
                    if x.startswith('[') and x.endswith(']'):
                        parsed = ast.literal_eval(x)
                        if isinstance(parsed, list):
                            cleaned_data.extend([float(i) for i in parsed])
                        else:
                            cleaned_data.append(float(parsed))
                    else:
                        cleaned_data.append(float(x))
            except (ValueError, SyntaxError):
                continue
                
        data = np.array(cleaned_data)
        shapiro_test(data, pipeline, eval)

def perform_normality_tests():
    df_map = {
        Metric.FAITHFULNESS: pd.read_csv(DATA_DIR / "faithfulness_scores.csv"),
        Metric.ANSWER_RELEVANCY: pd.read_csv(DATA_DIR / "answer_relevancy_scores.csv"),
        Metric.CONTEXT_RECALL: pd.read_csv(DATA_DIR / "context_recall_scores.csv"),
    }

    for eval, df in df_map.items():
        compute_normality_test(df, eval)
    generate_qq_grids()
    