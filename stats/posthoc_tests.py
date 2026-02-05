from operator import is_
from sched import scheduler
import pandas as pd
import os
from classes import *
from config import NUM_TRIALS, ALPHA
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from stats.visualizer_generator import generatePValueMatrix

# Define top-level directory (one level up from current script)
TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define target folder inside top-level directory
IO_DIR = os.path.join(TOP_LEVEL_DIR, "experimental-data")

def computePostHocTestForMetric(results, is_normal: bool, metric: Metric, workload: Workload):
    geometric_mean_df = pd.read_csv(os.path.join(IO_DIR, "geometric_means.csv"))

    # Build scheduler/result lists robustly so we don't fail when lengths differ
    sched_list = []
    res_list = []
    for scheduler in list(Scheduler):
        vals = results.get(scheduler.value, [])
        # ensure list
        if not isinstance(vals, list):
            try:
                vals = list(vals)
            except Exception:
                vals = [vals]
        if len(vals) != NUM_TRIALS:
            print(f"Warning: scheduler '{scheduler.value}' has {len(vals)} samples (expected {NUM_TRIALS}).")
        sched_list.extend([scheduler.value] * len(vals))
        res_list.extend(vals)

    if len(sched_list) == 0:
        raise ValueError("No result data available for post-hoc test")

    df = pd.DataFrame({"scheduler": sched_list, "results": res_list})
    posthoc_df = pd.DataFrame()
    print(f"is_normal: {is_normal}")
    
    scheduler1_mean, scheduler2_mean = 0, 0
    if is_normal:
        test = PostHocTest.TUKEY_HSD
        # pairwise_tukeyhsd returns a TukeyHSDResults object; pass that
        # object into tukey_to_dataframe which will handle extracting
        # the summary table.
        tukey_res = pairwise_tukeyhsd(endog=df['results'], groups=df['scheduler'])
        posthoc_df = tukey_to_dataframe(tukey_res)
        for i, row in posthoc_df.iterrows():
            if row['p-adj'] >= ALPHA:
                continue
            scheduler1 = row['group1']
            scheduler2 = row['group2']
            scheduler1_mean = geometric_mean_df.loc[(geometric_mean_df['scheduler'] == scheduler1) & (geometric_mean_df['metric'] == metric.value.lower().replace(" ", "_")) & (geometric_mean_df['workload'] == workload.value), 'gmean'].values[0]
            scheduler2_mean = geometric_mean_df.loc[(geometric_mean_df['scheduler'] == scheduler2) & (geometric_mean_df['metric'] == metric.value.lower().replace(" ", "_")) & (geometric_mean_df['workload'] == workload.value), 'gmean'].values[0] # type: ignore
            if metric in [Metric.AWT, Metric.ART]:
                if scheduler1_mean < scheduler2_mean and metric:
                    winner = scheduler1
                    loser = scheduler2
                else:
                    winner = scheduler2
                    loser = scheduler1
            else:
                if scheduler1_mean < scheduler2_mean:
                    winner = scheduler2
                    loser = scheduler1
                else:
                    winner = scheduler1
                    loser = scheduler2
            with open(os.path.join(IO_DIR, "posthoc_test_results.csv"), 'a', newline='') as f:
                writer = pd.DataFrame({
                    "workload": [workload.value],
                    "metric": [metric.value.lower().replace(" ", "_")],
                    "loser": [loser],
                    "winner": [winner],
                })
                writer.to_csv(f, header=f.tell()==0, index=False)
    else:
        test = PostHocTest.DUNN
        posthoc_df = posthoc_dunn(df, val_col='results', group_col='scheduler', p_adjust='bonferroni')
        for i, scheduler1 in enumerate(posthoc_df.columns):
            for scheduler2 in list(posthoc_df.index)[i+1:]:
                if posthoc_df.loc[scheduler2, scheduler1].item() >= ALPHA: # type: ignore
                    continue
                scheduler1_mean = geometric_mean_df.loc[(geometric_mean_df['scheduler'] == scheduler1) & (geometric_mean_df['metric'] == metric.value.lower().replace(" ", "_")) & (geometric_mean_df['workload'] == workload.value), 'gmean'].values[0]
                scheduler2_mean = geometric_mean_df.loc[(geometric_mean_df['scheduler'] == scheduler2) & (geometric_mean_df['metric'] == metric.value.lower().replace(" ", "_")) & (geometric_mean_df['workload'] == workload.value), 'gmean'].values[0] # type: ignore
                
                if metric in [Metric.AWT, Metric.ART]:
                    if scheduler1_mean < scheduler2_mean and metric:
                        winner = scheduler1
                        loser = scheduler2
                    else:
                        winner = scheduler2
                        loser = scheduler1
                else:
                    if scheduler1_mean < scheduler2_mean:
                        winner = scheduler2
                        loser = scheduler1
                    else:
                        winner = scheduler1
                        loser = scheduler2
                
                with open(os.path.join(IO_DIR, "posthoc_test_results.csv"), 'a', newline='') as f:
                    writer = pd.DataFrame({
                        "workload": [workload.value],
                        "metric": [metric.value.lower().replace(" ", "_")],
                        "loser": [loser],
                        "winner": [winner],
                    })
                    writer.to_csv(f, header=f.tell()==0, index=False)
        
    
    print(posthoc_df)
    generatePValueMatrix(df=posthoc_df, test=test, metric=metric, workload=workload)
    
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
    for workload in list(Workload):
        for metric in list(Metric):
            results = {}
            hypothesis_test_p_value = hypothesis_test_df.loc[(hypothesis_test_df['workload'] == workload.value) & (hypothesis_test_df['metric'] == metric.value.lower().replace(" ", "_")),'p_value'].values[0]
            if hypothesis_test_p_value >= 0.05:
                print(f"No significant difference found for {workload.value} - {metric.value} pair; skipping post-hoc test.")
                continue  # No need for post-hoc test if not significant
            print(f"Significant difference found for {workload.value} - {metric.value} pair; performing post-hoc test.")
            normality_test_results = []
            for scheduler in list(Scheduler):
                normality_test_results.append(normality_test_df.loc[(normality_test_df['workload'] == workload.value) & (normality_test_df['metric'] == metric.value.lower().replace(" ", "_")) & (normality_test_df['scheduler'] == scheduler.value),'normality'].values[0])
                scheduler_data = pd.read_csv(os.path.join(IO_DIR, f"{scheduler.value.lower()}_results.csv"))
                metric_data = scheduler_data.loc[scheduler_data['workload'] == workload.value, metric.value.lower().replace(" ", "_")].values
                results[scheduler.value] = metric_data.tolist()
            
            is_normal = all(nr == "NORMAL" for nr in normality_test_results)
            computePostHocTestForMetric(results, is_normal, metric, workload)