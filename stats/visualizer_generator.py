import matplotlib.pyplot as plt
import scipy.stats as scipy_stats
import numpy as np
import seaborn as sns
import pandas as pd
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import math
import os
import sys
import matplotlib.patheffects as pe

# Define top-level directory (one level up from current script)
TOP_LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Append top-level directory to sys.path BEFORE importing from it
sys.path.append(TOP_LEVEL_DIR)

from classes import Metric, Scheduler, Workload, PostHocTest
from config import ALPHA

# Define target folder inside top-level directory
OUTPUT_DIR = os.path.join(TOP_LEVEL_DIR, "visualizations")

def generate_qq_grid_for_metric(metric, df_map):
    """
    Generates a grid of Q-Q plots for each scheduler × workload combination for a given metric,
    runs Shapiro-Wilk normality test on each, and displays the p-value in the subplot.
    Saves the final grid in ../visualizations/ relative to this file.
    
    Args:
        metric: Metric enum value (e.g., Metric.FAIRNESS)
        df_map: dict {Scheduler: DataFrame}
        workloads: list of Workload enum values
    """

    # Determine output folder (visualizations/)
    workloads = list(Workload)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    out_folder = os.path.join(project_root, "visualizations")
    os.makedirs(out_folder, exist_ok=True)

    # Grid layout: rows = number of schedulers, columns = number of test categories
    schedulers = list(df_map.keys())
    nrows, ncols = len(schedulers), len(workloads)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

    for i, scheduler in enumerate(schedulers):
        df = df_map[scheduler]
        for j, workload in enumerate(workloads):
            ax = axes[i, j]

            # Extract values for this metric/workload/scheduler
            values = df.loc[df['workload'] == workload.value, metric.value.lower().replace(" ", "_")].dropna().to_numpy()

            # Skip empty datasets
            if len(values) == 0:
                ax.set_visible(False)
                continue

            # Run Shapiro-Wilk test
            if len(values) < 3:
                p_value = np.nan
                normality = "n<3"
            else:
                _, p_value = scipy_stats.shapiro(values)
                normality = "NORMAL" if p_value > ALPHA else "NOT NORMAL"

            # Generate Q-Q plot
            # Standardize sample values for visual comparability
            std = np.std(values)
            if std == 0:
                z_values = np.zeros_like(values)  # avoid division by zero
            else:
                z_values = (values - np.mean(values)) / std

            # Q-Q using standardized values
            osm, osr = scipy_stats.probplot(z_values, dist="norm")[0]

            print(f"Plotting QQ plot for Workload: {workload.value}, Scheduler: {scheduler.value}, Metric: {metric.value}")

            ax.scatter(osm, osr, s=10, color='black')

            # Optional reference diagonal line
            ax.plot([osm.min(), osm.max()], [osm.min(), osm.max()], 'r--', linewidth=0.8)

            # Title and axes
            ax.set_title(f"{scheduler.value} | {workload.value}\n n={len(values)}", fontsize=9)
            ax.set_xlabel("Theoretical Quantiles", fontsize=8)
            ax.set_ylabel("Sample Quantiles", fontsize=8)

            # Display p-value
            if not np.isnan(p_value):
                ax.text(0.95, 0.05, f"p={p_value:.3g}\n{normality}", ha='right', va='bottom', transform=ax.transAxes, fontsize=7)
            else:
                ax.text(0.95, 0.05, normality, ha='right', va='bottom', transform=ax.transAxes, fontsize=7)

    # Layout and save
    plt.suptitle(f"Q-Q Plots for {metric.value}", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    filename = metric.value.lower().replace(" ", "_")
    out_path = os.path.join(out_folder, f"normality_qq_grid_{filename}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved Q-Q grid for {metric.value} at: {out_path}")

def generateBoxPlotGrids(fcfs_df, rr_df, osjfdrr_df, srr_df, kdts_df):
    """
    Generate one grid of boxplots per workload. Each grid compares, for each metric,
    the distributions across the five schedulers (FCFS, RR, oSJFDRR, SRR, KDTS).

    Args:
        fcfs_df, rr_df, osjfdrr_df, srr_df, kdts_df : pandas.DataFrame
            DataFrames for each scheduler. Each must have a 'workload' column and
            columns named by metric.value (or the strings you pass in metrics).
        metrics : iterable of Enum or str
            Iterable of metric enum members (or objects with .value) or strings describing
            the metric column names to plot (order will be used).
        workloads : iterable of Enum or str
            Iterable of Workload enum members (or objects with .value) or strings for
            each workload for which a separate grid will be produced.
        out_folder : str or None
            If None, the function will save to ../visualizations relative to this file.
            Otherwise it will save to the provided path (created if necessary).

    Returns:
        list of str : paths of saved image files
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    out_folder = os.path.join(project_root, "visualizations")
    os.makedirs(out_folder, exist_ok=True)
    # Map friendly scheduler labels to dataframes (preserves order)
    df_map = [
        ("FCFS", fcfs_df),
        ("RR", rr_df),
        ("oSJFDRR", osjfdrr_df),
        ("SRR", srr_df),
        ("KDTS", kdts_df)
    ]
    scheduler_labels = [label for label, _ in df_map]

    saved_paths = []

    # Normalize metrics and workloads to strings (support Enum members)
    metric_names = [m.value.lower().replace(" ", "_") for m in list(Metric)]
    workload_names = [t.value for t in list(Workload)]
    scheduler_names = [s.name for s in list(Scheduler)]
    
    
    for metric in metric_names:
        n_workloads = len(workload_names)
        # layout: roughly square
        
        ncols = 2
        nrows = 2
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

        for idx, workload in enumerate(workload_names):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            # gather data arrays for each scheduler for the current metric & workload
            data_list = []
            sample_sizes = []
            for _, df in df_map:
                vals = df.loc[df['workload'] == workload, metric].dropna().to_numpy()
                data_list.append(vals if len(vals) > 0 else np.array([]))
                sample_sizes.append(len(vals))

            # If every scheduler has no data for this metric/workload, show placeholder
            if all(len(arr) == 0 for arr in data_list):
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(metric.upper().replace("_", " "), fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Create boxplot
            bp = ax.boxplot(data_list, labels=[f"{lab}\n(n={n})" for lab, n in zip(scheduler_labels, sample_sizes)],
                            patch_artist=True, showfliers=True)

            # Annotate median values above each box for clarity
            medians = []
            for i, d in enumerate(data_list):
                if len(d) == 0:
                    medians.append(np.nan)
                else:
                    medians.append(np.nanmedian(d))

            for i, m in enumerate(medians):
                if not np.isnan(m):
                   ax.text(
                        i + 1,
                        m,
                        f"{m:.3g}",
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        color='black',
                        weight='bold',
                        path_effects=[path_effects.Stroke(linewidth=0.5, foreground='yellow'), path_effects.Normal()]
                    )

            ax.set_title(workload.upper().replace("_", " "), fontsize=10)
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

        # Hide any unused subplots
        total_subplots = nrows * ncols
        for unused_idx in range(n_workloads, total_subplots):
            r = unused_idx // ncols
            c = unused_idx % ncols
            axes[r][c].axis('off')

        # Move the suptitle closer to the subplots by allowing subplots to
        # occupy more vertical space (rect top closer to 1.0) and placing the
        # suptitle slightly lower. This reduces the gap between the title and
        # the first row of boxplots.
        plt.suptitle(f"Boxplot Grid — Metric: {metric.upper().replace("_", " ")}", fontsize=14, y=0.98)
        plt.tight_layout(rect=(0, 0, 1, 0.99))
        
        out_path = os.path.join(out_folder, f"hypothesis_boxgrid_{metric.lower()}.png")
        print(f"[INFO] Saving boxplot grids to: {out_path}")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths

def generatePValueMatrix(df: pd.DataFrame, test: PostHocTest, metric: Metric, workload: Workload):
    workload = workload.value
    if test == PostHocTest.DUNN:
        matrix = df
        test_name_label = "Dunn's"
        test_name_file = "dunns"
    else:
        matrix = make_full_tukey_matrix(df)
        test_name_label = "Tukey's HSD"
        test_name_file = "tukeyhsd"
        
    colors = [(0.8, 0.0, 0.0),  # dark red
            (0.95, 0.95, 0.95),  # light gray (around alpha)
            (0.2, 0.2, 0.8)]  # blue
    custom_cmap = LinearSegmentedColormap.from_list("sigmap", colors, N=1024)

    # --- Force a sharp transition around alpha ---
    norm = TwoSlopeNorm(vmin=0, vcenter=ALPHA, vmax=1)

    # Ensure the matrix contains numeric p-values. Coerce if possible and
    # raise a clear error if non-numeric entries are present (helps catch
    # upstream mistakes where a DataFrame of labels was passed accidentally).
    raw_matrix = matrix.copy()
    try:
        numeric_matrix = matrix.astype(float)
    except Exception:
        numeric_matrix = matrix.apply(pd.to_numeric, errors='coerce')
        # find first non-numeric entries (if any) in the original matrix
        mask = numeric_matrix.isna() & ~raw_matrix.isna()
        if mask.values.any():
            coords = list(zip(*np.where(mask.values)))
            sample = []
            for (r, c) in coords[:6]:
                sample.append({'row': raw_matrix.index[r], 'col': raw_matrix.columns[c], 'value': raw_matrix.iat[r, c]})
            raise ValueError(
                "Post-hoc p-value matrix contains non-numeric entries. "
                f"First offending cells: {sample}.\nEnsure you pass a numeric matrix of p-values to generatePValueMatrix."
            )

    # --- Plot heatmap ---
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        numeric_matrix,
        annot=True,
        fmt=".3f",
        cmap=custom_cmap,
        norm=norm,
        cbar_kws={'label': f'p-value (α={ALPHA})'}
    )
    metric_name_label = metric.name.capitalize() if metric != Metric.CPU_UTILIZATION else "CPU Utilization"
    metric_name_file = metric.name.lower() if metric != Metric.CPU_UTILIZATION else "cpu_utilization"
    plt.title(f"{test_name_label} Test for {metric_name_label} - {workload} Workload")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"posthoc_heatmap_{metric_name_file}_{test_name_file}_{workload.lower()}.png"), dpi=300)
    
def make_full_tukey_matrix(tukey_df):
    # Find the columns that contain the group names
    possible_group_cols = [col for col in tukey_df.columns if 'group' in col.lower()]
    if len(possible_group_cols) < 2:
        raise ValueError("DataFrame must contain two group columns")

    g1_col, g2_col = possible_group_cols[:2]

    # Find p-value column
    # Prefer an explicit 'p-adj' or similar column name; avoid accidentally
    # matching columns like 'meandiff' which also contain the letter 'p'.
    pval_candidates = [col for col in tukey_df.columns if 'p-adj' in col.lower() or 'p_adj' in col.lower() or col.lower() == 'p' or 'p-value' in col.lower() or 'p_value' in col.lower()]
    if not pval_candidates:
        # Fallback: any column containing 'p' but not 'meandiff'
        pval_candidates = [col for col in tukey_df.columns if ('p' in col.lower()) and ('meandiff' not in col.lower())]
    if not pval_candidates:
        raise ValueError("Could not find a p-value column in the Tukey results DataFrame")
    pval_col = pval_candidates[0]

    groups = np.unique(tukey_df[[g1_col, g2_col]].values)
    matrix = pd.DataFrame(1.0, index=groups, columns=groups)

    for _, row in tukey_df.iterrows():
        g1, g2, pval = row[g1_col], row[g2_col], row[pval_col]
        matrix.loc[g1, g2] = pval
        matrix.loc[g2, g1] = pval

    return matrix

# Example dataset
np.random.seed(0)
data = pd.DataFrame({
    'group': np.repeat(['A', 'B', 'C', 'D'], 20),
    'value': np.concatenate([
        np.random.normal(5, 1, 20),
        np.random.normal(6, 1, 20),
        np.random.normal(8, 1, 20),
        np.random.normal(9, 1, 20)
    ])
})

