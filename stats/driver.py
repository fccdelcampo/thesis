import os
import sys
import pandas as pd
import numpy as np

from stats.negative_scoring_tally import tallyNegativeScoringResults

# Define top-level directory (one level up from current script)
TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Append top-level directory to sys.path BEFORE importing
sys.path.append(TOP_LEVEL_DIR)

from classes import *
from stats.normality_tests import performNormalityTests
from stats.hypothesis_tests import performHypothesisTests
from stats.posthoc_tests import performPostHocTests

# Define target folder inside top-level directory
IO_DIR = os.path.join(TOP_LEVEL_DIR, "experimental-data")

def generateDataFrames():
    dfs = {}
    for scheduler in list(Scheduler):
        scheduler_data = pd.read_csv(os.path.join(IO_DIR, f"{scheduler.value.lower()}_results.csv"))
        dfs[scheduler] = scheduler_data
    return dfs
    

def ensureCsvHeaders():
    """Ensure all stats CSV files have their proper headers before writing data."""
    headers = {
        "normality_test_results.csv": "scheduler,metric,workload,p_value,normality\n",
        "hypothesis_test_results.csv": "workload,metric,other_scheduler,p_value,interpretation\n",
        "posthoc_test_results.csv": "workload,metric,loser,winner\n",
        "geometric_means.csv": "metric,scheduler,workload,gmean\n",
        "negative_scoring_results.csv": "metric,workload,wins,losses,net_score\n"
    }
    
    for filename, header in headers.items():
        path = os.path.join(IO_DIR, filename)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(header)

def performStatisticalTests():
    ensureCsvHeaders()
    
    dfs = generateDataFrames()
    fcfs_df = dfs[Scheduler.FCFS]
    rr_df = dfs[Scheduler.RR]
    osjfdrr_df = dfs[Scheduler.oSJFDRR]
    srr_df = dfs[Scheduler.SRR]
    kdts_df = dfs[Scheduler.KDTS]
    
    # print("Starting Normality Tests...")
    # performNormalityTests(fcfs_df, rr_df, osjfdrr_df, srr_df, kdts_df)
    
    print("Starting Hypothesis Tests...")
    performHypothesisTests(fcfs_df, rr_df, osjfdrr_df, srr_df, kdts_df)
    
    print("Starting Negative Scoring Tally... ")    
    tallyNegativeScoringResults()        
    # print("Starting Post Hoc Tests...")
    # performPostHocTests()
    
    print("Statistical Tests Completed.")
    
def generateArtificialData():
    # For <scheduler>_results.csv files
    # workload,awt,art,throughput,cpu_utilization
    print("Generating artificial data...")
    for scheduler in list(Scheduler):
        for workload in list(Workload):
            for i in range(120):
                with open(os.path.join(IO_DIR, f"{scheduler.value.lower()}_results.csv"), 'a', newline='') as f:
                    writer = pd.DataFrame({
                        "workload": [workload.value],
                        "awt": [np.random.uniform(0.7, 1.0)],
                        "art": [np.random.uniform(50, 150)],
                        "throughput": [np.random.uniform(10, 50)],
                        "cpu_utilization": [np.random.uniform(70, 100)]
                    })
                    writer.to_csv(f, header=f.tell()==0, index=False)
    print("Done generating artificial data.")
    
def informal_tests():
    print("================Starting informal tests...================")
    generateArtificialData()
    performStatisticalTests()
  
if __name__ == "__main__":  
    informal_tests()