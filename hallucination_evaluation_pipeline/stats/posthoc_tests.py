import pandas as pd
import numpy as np
import os
import csv
from config import *
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import mcnemar
# from stats.visualizer_generator import generatePValueMatrix
from scipy.stats import rankdata
import visualizer_generator

# Define top-level directory (one level up from current script)
TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# RESULTS_DIR = ROOT / "hallucination_evaluation_results"
sys.path.append(str(ROOT))
# Define target folder inside top-level directory
IO_DIR = ROOT / "hallucination_evaluation_results"

def log_posthoc_result(eval_name: str,loser: str, winner: str, p_value: float):
    with open(IO_DIR / "posthoc_test_results.csv", 'a', newline='') as f:
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

def compute_posthoc_test_for_metric(results, is_normal: bool, eval: Metric | Benchmark):
    print(f"RESULTS:\n{results}")
    if eval in CONTINUOUS_EVALS:
        if is_normal:
            df = pd.DataFrame(
                {
                    'score': results["tuned"] + results["graphrag"] + results["prolog_graphrag"],
                    'group': np.repeat(['tuned', 'graphrag', "prolog_graphrag"], repeats=NUM_TEST_CASES)
                }
            )
            result_df = pairwise_tukeyhsd(endog=df["score"], groups=df["group"], alpha=ALPHA)
            significant_rows = df.loc[df["reject"], ["group1", "group2", "p-adj"]].values.tolist()
            print(f"SIG ROWS:\n {significant_rows}")
            for row in significant_rows:
                p_value = row[2]
                all_scores = results[row[0]] + results[row[1]]
                ranks = rankdata(all_scores)
                pipeline1_mean_rank = np.mean(ranks[:NUM_TEST_CASES])
                pipeline2_mean_rank = np.mean(ranks[NUM_TEST_CASES:])
                winner = row[0] if pipeline1_mean_rank > pipeline2_mean_rank else row[1]
                loser = row[1] if pipeline1_mean_rank < pipeline2_mean_rank else row[0]
                log_posthoc_result(eval_name=eval.value[1], loser=loser, winner=winner, p_value=p_value)
        else:
            df = pd.DataFrame(
                {
                    'score': results["tuned"] + results["graphrag"] + results["prolog_graphrag"],
                    'group': np.repeat(['tuned', 'graphrag', "prolog_graphrag"], repeats=NUM_TEST_CASES)
                }
            )
            result_df = posthoc_dunn(df, val_col="score", group_col="group")
            print("DUNN'S TEST DATAFRAME:\n", result_df)
            for i, pipeline1 in enumerate(list(Pipeline)):
                for pipeline2 in list(Pipeline)[i+1:]:
                    pipeline1_name = pipeline1.value[1]
                    pipeline2_name = pipeline2.value[1]
                    print(f"{pipeline1_name} | {pipeline2_name}")
                    p_value = result_df.loc[pipeline1_name, pipeline2_name]
                    if p_value < ALPHA:
                        all_scores = results[pipeline1_name] + results[pipeline2_name]
                        ranks = rankdata(all_scores)
                        pipeline1_mean_rank = np.mean(ranks[:NUM_TEST_CASES])
                        pipeline2_mean_rank = np.mean(ranks[NUM_TEST_CASES:])
                        winner, loser = (pipeline1_name, pipeline2_name) if pipeline1_mean_rank > pipeline2_mean_rank else (pipeline2_name, pipeline1_name)
                        log_posthoc_result(eval_name=eval.value[1], loser=loser, winner=winner, p_value=p_value)
    else:
        for pipeline1 in list(Pipeline):
            for pipeline2 in list(Pipeline):
                if pipeline1 == pipeline2:
                    continue
                contingency_table = create_contingency_table(results=results, pipeline1=pipeline1, pipeline2=pipeline2)
                result = mcnemar(contingency_table, exact=True, correction=False)
                stat, p_value = result.statistic, result.pvalue
                if p_value < ALPHA:
                    if contingency_table[0, 1] > contingency_table[1, 0]:
                        winner = pipeline1.lower().replace("-", "_").replace(" ", "_").replace("pipeline", "")
                        loser = pipeline2.lower().replace("-", "_").replace(" ", "_").replace("pipeline", "")
                    else:
                        winner = pipeline2.lower().replace("-", "_").replace(" ", "_").replace("pipeline", "")
                        loser = pipeline1.lower().replace("-", "_").replace(" ", "_").replace("pipeline", "")
        
    # generatePValueMatrix(df=posthoc_df, test=test, metric=eval)

def create_contingency_table(results: dict, pipeline1: Pipeline, pipeline2: Pipeline):
    if pipeline1 == pipeline2:
        raise ValueError(f"Pipelines 1 and 2 ({pipeline1.value}) are the same.")
    if results[pipeline1.value].size != results[pipeline2.value].size:
        raise ValueError(f"Binary data of {pipeline1.value} and {pipeline2.value} are unequal in size.")
    both_correct, both_incorrect, pipeline1_only_correct, pipeline2_only_correct = 0, 0, 0, 0
    for test_case_id in range(results[pipeline1.value].size):
        pipeline1_score = results[pipeline1][test_case_id]
        pipeline2_score = results[pipeline2][test_case_id]
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
    hypothesis_test_df = pd.read_csv(IO_DIR / "hypothesis_test_results.csv")
    normality_test_df = pd.read_csv(IO_DIR / "normality_test_results.csv")
    print(normality_test_df)
    for eval in list(Metric) + list(Benchmark):
        results = {}
        hypothesis_test_interpretation = hypothesis_test_df.loc[hypothesis_test_df['eval_name'] == eval.value[1],'interpretation'].values[0]
        if hypothesis_test_interpretation == "NON_SIGNIFICANT":
            print(f"No significant difference found for{eval.value} pair; skipping post-hoc test.")
            continue  # No need for post-hoc test if not significant
        
        normality_test_results = []
        is_normal = False
        
        for pipeline in list(Pipeline):
            if eval in CONTINUOUS_EVALS:
                eval_name = eval.value[1]
                pipeline_name = pipeline.value[1]
                print(f"Eval name: {eval_name} | Pipeline name: {pipeline_name}")
                normality = normality_test_df.loc[(normality_test_df['eval_name'] == eval_name) & (normality_test_df['pipeline'] == pipeline_name),'normality'].values[0]
                normality_test_results.append(normality)
            eval_df = pd.read_csv(IO_DIR / f'{eval.value[1]}_scores.csv')
            col_name = pipeline.value[1] + "_score"
            pipeline_data = eval_df[col_name]
            results[pipeline.value[1]] = pipeline_data.tolist()
            
        if eval in CONTINUOUS_EVALS:
            is_normal = all(nr == "NORMAL" for nr in normality_test_results)
        
        compute_posthoc_test_for_metric(results, is_normal, eval)