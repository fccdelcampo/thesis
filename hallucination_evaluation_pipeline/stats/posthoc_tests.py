import pandas as pd
import numpy as np
import os
import csv
from visualizer_generator import *
from config import *
STATS_AVAILABLE = True
# try:
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import rankdata
# except ImportError:
#     STATS_AVAILABLE = False
#     print("Warning: statsmodels, scikit-posthocs, or scipy not found. Post-hoc tests will be skipped.")
# try:
#     from .config import *
#     from . import visualizer_generator
# except ImportError:
#     try:
#         from config import *

import sys
from pathlib import Path

_current_dir = os.path.dirname(os.path.abspath(__file__))
# Determine the project root. Assuming this file is in 'project_root/hallucination_evaluation_pipeline/stats/'
# parents[0] is 'hallucination_evaluation_pipeline', parents[1] is 'project_root'
ROOT = Path(_current_dir).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# Define target folder inside top-level directory
DATA_DIR = ROOT / "hallucination_evaluation_results/data"

def log_posthoc_result(eval_name: str,loser: str, winner: str, p_value: float):
    with open(DATA_DIR / "posthoc_test_results.csv", 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["eval_name", "loser", "winner", "p_value"]
        )
        writer.writerow({
            "eval_name": eval_name,
            "loser": loser,
            "winner": winner,
            "p_value": p_value,
        })

from typing import Union

def compute_posthoc_test_for_metric(results, is_normal: bool, eval: Union[Metric, Benchmark]):
    # print(f"RESULTS:\n{results}")
    p_value_dict = {}
    if eval in CONTINUOUS_EVALS:
        if is_normal:
            n_samples = len(results["tuned"])
            if n_samples < 2:
                print(f"Insufficient samples ({n_samples}) for {eval.value[1]}. Skipping.")
                return
            df = pd.DataFrame(
                {
                    'score': results["tuned"] + results["graphrag"] + results["prolog_graphrag"],
                    'group': np.repeat(['tuned', 'graphrag', "prolog_graphrag"], repeats=n_samples)
                }
            )
            result_df = pairwise_tukeyhsd(endog=df["score"], groups=df["group"], alpha=ALPHA)
            significant_rows = result_df.loc[result_df["reject"], ["group1", "group2", "p-adj"]].values.tolist()
            rows = result_df.loc[result_df["reject"], ["group1", "group2", "p-adj"]].values.tolist()

            print(f"SIG ROWS:\n {significant_rows}")
            for row in rows:
                p_value = row[2]
                pipeline1_name, pipeline2_name = row[0], row[1]
                p_value_dict[(pipeline1.value[0].replace(" Pipeline", ""), pipeline2.value[0].replace(" Pipeline", ""))] = p_value
                if row in significant_rows:
                    all_scores = results[pipeline1_name] + results[pipeline2_name]
                    ranks = rankdata(all_scores)
                    pipeline1_mean_rank = np.mean(ranks[:NUM_TEST_CASES])
                    pipeline2_mean_rank = np.mean(ranks[NUM_TEST_CASES:])
                    winner = pipeline1_name if pipeline1_mean_rank > pipeline2_mean_rank else pipeline2_name
                    loser = pipeline2_name if pipeline1_mean_rank < pipeline2_mean_rank else pipeline1_name
                    log_posthoc_result(eval_name=eval.value[1], loser=loser, winner=winner, p_value=p_value)
        else:
            n_samples = len(results["tuned"])
            if n_samples < 2:
                print(f"Insufficient samples ({n_samples}) for {eval.value[1]}. Skipping.")
                return
            df = pd.DataFrame(
                {
                    'score': results["tuned"] + results["graphrag"] + results["prolog_graphrag"],
                    'group': np.repeat(['tuned', 'graphrag', "prolog_graphrag"], repeats=n_samples)
                }
            )
            result_df = posthoc_dunn(df, val_col="score", group_col="group")
            # print("DUNN'S TEST DATAFRAME:\n", result_df)
            for i, pipeline1 in enumerate(list(Pipeline)):
                for pipeline2 in list(Pipeline)[i+1:]:
                    pipeline1_name = pipeline1.value[1]
                    pipeline2_name = pipeline2.value[1]
                    # print(f"{pipeline1_name} | {pipeline2_name}")
                    p_value = result_df.loc[pipeline1_name, pipeline2_name]
                    # print(f"pipeline1: {pipeline1}")
                    # print(f" pipeline1.value[0]: {pipeline1.value[0]}")
                    p_value_dict[((pipeline1.value[0]).replace(" Pipeline", ""), (pipeline2.value[0]).replace(" Pipeline", ""))] = p_value
                    if p_value < ALPHA:
                        all_scores = results[pipeline1_name] + results[pipeline2_name]
                        ranks = rankdata(all_scores)
                        current_n = len(results[pipeline1_name])
                        pipeline1_mean_rank = np.mean(ranks[:current_n])
                        pipeline2_mean_rank = np.mean(ranks[current_n:])
                        winner, loser = (pipeline1_name, pipeline2_name) if pipeline1_mean_rank > pipeline2_mean_rank else (pipeline2_name, pipeline1_name)
                        log_posthoc_result(eval_name=eval.value[1], loser=loser, winner=winner, p_value=p_value)
    else:
        for i, pipeline1 in enumerate(list(Pipeline)):
            for pipeline2 in list(Pipeline)[i+1:]:
                pipeline1_name = pipeline1.value[1]
                pipeline2_name = pipeline2.value[1]
                if pipeline1 == pipeline2:
                    continue
                contingency_table = create_contingency_table(results=results, pipeline1=pipeline1, pipeline2=pipeline2)
                result = mcnemar(contingency_table, exact=True, correction=False)
                stat, p_value = result.statistic, result.pvalue
                p_value_dict[(pipeline1.value[0].replace(" Pipeline", ""), pipeline2.value[0].replace(" Pipeline", ""))] = p_value
                if p_value < ALPHA:
                    if contingency_table[0, 1] > contingency_table[1, 0]:
                        winner = pipeline1_name
                        loser = pipeline2_name
                    else:
                        winner = pipeline2_name
                        loser = pipeline1_name
                    log_posthoc_result(eval_name=eval.value[1], loser=loser, winner=winner, p_value=p_value)
    generate_p_value_matrix(p_value_dict, eval)

def create_contingency_table(results: dict, pipeline1: Pipeline, pipeline2: Pipeline):
    if pipeline1 == pipeline2:
        raise ValueError(f"Pipelines 1 and 2 ({pipeline1.value}) are the same.")
    p1_key = pipeline1.value[1]
    p2_key = pipeline2.value[1]
    
    if len(results[p1_key]) != len(results[p2_key]):
        raise ValueError(f"Binary data of {p1_key} and {p2_key} are unequal in size.")
    
    both_correct, both_incorrect, pipeline1_only_correct, pipeline2_only_correct = 0, 0, 0, 0
    size = len(results[p1_key])
    
    for test_case_id in range(size):
        pipeline1_score = results[p1_key][test_case_id]
        pipeline2_score = results[p2_key][test_case_id]
        if pipeline1_score == pipeline2_score:
            if pipeline1_score == 1:
                both_correct += 1
            else:
                both_incorrect +=1
        else:
            if pipeline1_score == 1:
                pipeline1_only_correct += 1
            else:
                pipeline2_only_correct +=1
                
    contingency_table = np.array(
        [
            [both_correct, pipeline1_only_correct],
            [pipeline2_only_correct, both_incorrect]
        ]
    )

    return contingency_table

def perform_posthoc_tests():
    hypothesis_test_df = pd.read_csv(DATA_DIR / "hypothesis_test_results.csv")
    normality_test_df = pd.read_csv(DATA_DIR / "normality_test_results.csv")
    print(normality_test_df)
    for eval in list(Metric) + list(Benchmark):
        results = {}
        hypothesis_test_interpretation = hypothesis_test_df.loc[hypothesis_test_df['eval_name'] == eval.value[1],'interpretation'].values[0]
        if hypothesis_test_interpretation == "NON_SIGNIFICANT":
            print(f"No significant difference found for {eval.value[0]}; skipping post-hoc test.")
            continue  # No need for post-hoc test if not significant
        
        normality_test_results = []
        is_normal = False
        
        for pipeline in list(Pipeline):
            if eval in CONTINUOUS_EVALS:
                eval_name = eval.value[1]
                pipeline_name = pipeline.value[1]
                # print(f"Eval name: {eval_name} | Pipeline name: {pipeline_name}")
                row = normality_test_df.loc[(normality_test_df['eval_name'] == eval_name) & (normality_test_df['pipeline'] == pipeline_name),'normality']
                if row.empty:
                    print(f"Warning: No normality result for {eval_name} - {pipeline_name}. Assuming NOT normal.")
                    normality = "NOT_NORMAL"
                else:
                    normality = row.values[0]
                normality_test_results.append(normality)
            eval_df = pd.read_csv(DATA_DIR / f'{eval.value[1]}_scores.csv')
            col_name = pipeline.value[1] + "_score"
            pipeline_data = eval_df[col_name]
            results[pipeline.value[1]] = pipeline_data.tolist()
            
        if eval in CONTINUOUS_EVALS:
            is_normal = all(nr == "NORMAL" for nr in normality_test_results)
        
        compute_posthoc_test_for_metric(results, is_normal, eval)