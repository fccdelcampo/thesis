from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from classes import *
import numpy as np
import csv
import os

from stats.visualizer_generator import generateBoxPlotGrids

# Define top-level directory (one level up from current script)
TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define target folder inside top-level directory
IO_DIR = os.path.join(TOP_LEVEL_DIR, "experimental-data")

def computeHypothesisTestForMetric(fcfs_ndarray, rr_ndarray, osjfdrr_ndarray, srr_ndarray, kdts_ndarray, metric: Metric, workload: Workload):
    """
    Performs hypothesis testing (e.g., t-test or Mann-Whitney U test) for a given metric
    across different schedulers.

    Args:
        fcfs_ndarray (np.ndarray): Data for FCFS scheduler.
        rr_ndarray (np.ndarray): Data for RR scheduler.
        osjfdrr_ndarray (np.ndarray): Data for oSJFDRR scheduler.
        kdts_ndarray (np.ndarray): Data for KDTS scheduler.
        metric (Metric): The performance metric being tested.
    """
    other_schedulers = []
    print("Current metric: ", metric.value)
    match(metric):
        case Metric.AWT:
            other_schedulers = [(Scheduler.FCFS, fcfs_ndarray), (Scheduler.RR, rr_ndarray), (Scheduler.oSJFDRR, osjfdrr_ndarray), (Scheduler.SRR, srr_ndarray)]
        case Metric.ART:
            other_schedulers = [(Scheduler.FCFS, fcfs_ndarray)]
        case Metric.CPU_UTILIZATION:
            other_schedulers = [(Scheduler.RR, rr_ndarray), (Scheduler.oSJFDRR, osjfdrr_ndarray), (Scheduler.SRR, srr_ndarray)]
        case default:
            raise ValueError(f"Unsupported metric: {metric}")
    print(f"METRIC: {metric.value}, WORKLOAD: {workload.value}")
        
    # workload,metric,other_scheduler,p_value,interpretation
    comparisons = []   # list of dicts

    for other in other_schedulers:
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
                fieldnames=["workload", "metric", "other_scheduler", "p_value", "interpretation"]
            )
            writer.writerow({
                "workload": workload.value,
                "metric": metric.value.lower().replace(" ", "_"),
                "other_scheduler": c["other"],
                "p_value": final_p,
                "interpretation": interpretation
            })
                
        
def performHypothesisTests(fcfs_df, rr_df, osjfdrr_df, srr_df, kdts_df):
    df_map = {
        Scheduler.FCFS: fcfs_df,
        Scheduler.RR: rr_df,
        Scheduler.oSJFDRR: osjfdrr_df,
        Scheduler.SRR: srr_df,
        Scheduler.KDTS: kdts_df
    }

    # for workload in list(Workload):
    #     for metric in list(Metric):
    #         fcfs_data = np.array(df_map[Scheduler.FCFS].loc[df_map[Scheduler.FCFS]['workload'] == workload.value, metric.value.lower().replace(" ", "_")])
    #         rr_data = np.array(df_map[Scheduler.RR].loc[df_map[Scheduler.RR]['workload'] == workload.value, metric.value.lower().replace(" ", "_")])
    #         osjfdrr_data = np.array(df_map[Scheduler.oSJFDRR].loc[df_map[Scheduler.oSJFDRR]['workload'] == workload.value, metric.value.lower().replace(" ", "_")])
    #         srr_data = np.array(df_map[Scheduler.oSJFDRR].loc[df_map[Scheduler.oSJFDRR]['workload'] == workload.value, metric.value.lower().replace(" ", "_")])
    #         kdts_data = np.array(df_map[Scheduler.KDTS].loc[df_map[Scheduler.KDTS]['workload'] == workload.value, metric.value.lower().replace(" ", "_")])

    #         computeHypothesisTestForMetric(
    #             fcfs_ndarray=fcfs_data,
    #             rr_ndarray=rr_data,
    #             osjfdrr_ndarray=osjfdrr_data,
    #             srr_ndarray=srr_data,
    #             kdts_ndarray=kdts_data,
    #             metric=metric,
    #             workload=workload
    #         )
            
    generateBoxPlotGrids(
        fcfs_df=fcfs_df,
        rr_df=rr_df,
        osjfdrr_df=osjfdrr_df,
        srr_df=srr_df,
        kdts_df=kdts_df
    )
    