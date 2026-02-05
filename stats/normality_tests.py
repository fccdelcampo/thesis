import os
import sys
import numpy as np
import csv

# Scipy imports FIRST (before wildcard import)
from scipy.stats import gmean, shapiro

# Define top-level directory (one level up from current script)
TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append top-level directory to sys.path BEFORE importing
sys.path.append(TOP_LEVEL_DIR)

# Import classes and config (but this brings in stats from standard library)
from classes import Process, ProcessGroup, Metric, Scheduler, Workload
from config import ALPHA
from .visualizer_generator import generate_qq_grid_for_metric

# Define target folder inside top-level directory
IO_DIR = os.path.join(TOP_LEVEL_DIR, "experimental-data")

def shapiro_test(data: np.ndarray, scheduler, metric, workload):
    if len(data) < 3:
        return np.nan, "TOO_SMALL_N"
    _, p_value = shapiro(data)
    geometric_mean = gmean(data)
    normality = "NORMAL" if p_value > ALPHA else "NOT_NORMAL"

    # Write geometric mean results to CSV
    gmean_row = {
        "metric": metric.value.lower().replace(" ", "_"),
        "scheduler": scheduler.value,
        "workload": workload.value,
        "gmean": geometric_mean,
    }

    with open(os.path.join(IO_DIR, "geometric_means.csv"), 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=gmean_row.keys())
        writer.writerow(gmean_row)
        
    # Write normality test results to CSV
    normality_row = {
        "scheduler": scheduler.value,
        "metric": metric.value.lower().replace(" ", "_"),
        "workload": workload.value,
        "p_value": p_value,
        "normality": normality
    }

    with open(os.path.join(IO_DIR, "normality_test_results.csv"), 'a') as f:
        writer = csv.DictWriter(f, fieldnames=normality_row.keys())
        writer.writerow(normality_row)

def computeNormalityTest(df, scheduler):
    for metric in list(Metric):
        for workload in list(Workload):
            data = np.array(df.loc[df['workload'] == workload.value, metric.value.lower().replace(" ", "_")])
            shapiro_test(data, scheduler, metric, workload)

def performNormalityTests(fcfs_df, rr_df, osjfdrr_df, srr_df, kdts_df):
    df_map = {
        Scheduler.FCFS: fcfs_df,
        Scheduler.RR: rr_df,
        Scheduler.oSJFDRR: osjfdrr_df,
        Scheduler.SRR: srr_df,
        Scheduler.KDTS: kdts_df
    }

    for scheduler, df in df_map.items():
        computeNormalityTest(df, scheduler)

    for metric in Metric:
        generate_qq_grid_for_metric(
            metric=metric,
            df_map=df_map
        )
    