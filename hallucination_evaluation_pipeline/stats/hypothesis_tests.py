import sys
import os
import numpy as np
import pandas as pd
import csv
from pathlib import Path

# Project root and stats directory setup
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
# Determine the project root. Assuming this file is in 'project_root/stats/'
# parents[0] is 'stats', parents[1] is 'project_root'
ROOT = Path(_current_dir).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from config import ALPHA, Metric, Benchmark, CONTINUOUS_EVALS, DISCRETE_EVALS
    import visualizer_generator
    from visualizer_generator import generate_box_plot_grids, generate_bar_graph_grid
except ImportError:
    # Fallback for relative imports if the script is run as a module
    try:
        from .config import ALPHA, Metric, Benchmark, CONTINUOUS_EVALS, DISCRETE_EVALS
        from . import visualizer_generator
        from .visualizer_generator import generate_box_plot_grids, generate_bar_graph_grid
    except ImportError:
        # Fallback for when 'stats' is a package and config/visualizer_generator are siblings
        from stats.config import ALPHA, Metric, Benchmark, CONTINUOUS_EVALS, DISCRETE_EVALS
        from stats import visualizer_generator
        from stats.visualizer_generator import generate_box_plot_grids, generate_bar_graph_grid


STATS_AVAILABLE = True
try:
    from scipy.stats import kruskal, mannwhitneyu
    from statsmodels.stats.contingency_tables import cochrans_q
except ImportError:
    STATS_AVAILABLE = False
    print("Warning: scipy or statsmodels not found. Hypothesis tests will be skipped.")

# Define target folder inside top-level directory
DATA_DIR = ROOT / "hallucination_evaluation_results/data"

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
        Metric.ANSWER_RELEVANCY: pd.read_csv(DATA_DIR / "answer_relevancy_scores.csv"),
        Metric.CONTEXT_RECALL: pd.read_csv(DATA_DIR / "context_recall_scores.csv"),
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
            
    generate_box_plot_grids()
    
    
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
        Benchmark.AI2_ARC: pd.read_csv(DATA_DIR / "ai2_arc_scores.csv"),
        Benchmark.OPENBOOKQA: pd.read_csv(DATA_DIR / "openbookqa_scores.csv"),
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
            
    generate_bar_graph_grid()

def perform_hypothesis_tests():
    perform_continuous_hypothesis_tests()
    perform_discrete_hypothesis_tests()