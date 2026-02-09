from operator import is_
from sched import pipeline
import pandas as pd
import os
from config import *
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from stats.visualizer_generator import generatePValueMatrix

# Define top-level directory (one level up from current script)
TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define target folder inside top-level directory
IO_DIR = os.path.join(TOP_LEVEL_DIR, "experimental_data")

def computePostHocTestForContinuousMetric(results, is_normal: bool, eval: Metric):
    if is_normal:
        results = posthoc_dunn(results)
    else:

    # generatePValueMatrix(df=posthoc_df, test=test, metric=eval)

def create_contingency_table(results: dict):
    for pipeline1 in list(Pipeline):
        for pipeline2 in list(Pipeline):
            if pipeline1==pipeline2:
                continue
            if results[pipeline1.value].size != results[pipeline2.value].size:
                raise ValueError(f"Binary data of {pipeline1.value} and {pipeline2.value} are unequal in size.")
            
            for test_case in range(results[pipeline1.value].size):
                pass

    
def tukey_to_dataframe(tukey_result):
    """
    Converts a statsmodels TukeyHSDResults object into a DataFrame compatible
    with `make_full_tukey_matrix`.
    
    Columns: ['group1', 'group2', 'meandiff', 'p-adj', 'lower', 'upper', 'reject']
    """
    # tukey_result may be either a statsmodels TukeyHSDResults object
    # (which exposes .summary()) or already a SimpleTable (the summary).
    if hasattr(tukey_result, 'summary'):
        summary = tukey_result.summary()
    else:
        summary = tukey_result

    # Convert summary table to DataFrame
    data = summary.data[1:]  # skip header row
    columns = summary.data[0]  # header row

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Ensure p-value column is float
    df['p-adj'] = df['p-adj'].astype(float)

    return df

def performPostHocTests():
    hypothesis_test_df = pd.read_csv(os.path.join(IO_DIR, "hypothesis_test_results.csv"))
    normality_test_df = pd.read_csv(os.path.join(IO_DIR, "normality_test_results.csv"))
    for eval in list(Metric) + list(Benchmark):
        results = {}
        hypothesis_test_interpretation = hypothesis_test_df.loc[(hypothesis_test_df['eval_name'] == eval.value.lower().replace(" ", "_")),'interpretation'].values[0]
        if hypothesis_test_interpretation == "NON_SIGNIFICANT":
            print(f"No significant difference found for{eval.value} pair; skipping post-hoc test.")
            continue  # No need for post-hoc test if not significant
        print(f"Significant difference found for {workload.value} - {metric.value} pair; performing post-hoc test.")
        
        normality_test_results = []
        
        for pipeline in list(Pipeline):
            if eval in CONTINUOUS_EVALS:
             normality_test_results.append(normality_test_df.loc[(normality_test_df['eval_name'] == eval.value.lower().replace(" ", "_")) & (normality_test_df['pipeline'] == pipeline.value.lower().replace("-", "_").replace(" ", "_").replace("pipeline", "")),'normality'].values[0])
            eval_df = pd.read_csv(rf"../../hallucination_evaluation_results/{eval.value.lower().replace(" ", "_")}_scores.csv")
            pipeline_data = eval_df[pipeline.value.lower().replace("-", "_").replace(" ", "_").replace("pipeline", "")]
            results[pipeline.value] = pipeline_data.tolist()
            
        if eval in CONTINUOUS_EVALS:
            is_normal = all(nr == "NORMAL" for nr in normality_test_results)
            computePostHocTestForContinuousMetric(results, is_normal, eval)
        else:
            computePostHocTestForDiscreteMetric(results, is_normal, eval)