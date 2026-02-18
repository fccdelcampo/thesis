import os
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.stats as scipy_stats
import numpy as np
import seaborn as sns
import pandas as pd

from config import ALPHA, CONTINUOUS_EVALS, DISCRETE_EVALS, Pipeline

# Project root and results/visualizations paths
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "hallucination_evaluation_results/data"
OUTPUT_DIR = ROOT / "hallucination_evaluation_results/visualizations"

import ast

def parse_value_robust(x):
    """Parses a single value robustly, handling stringified lists."""
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            x = x.strip()
            if x.startswith('[') and x.endswith(']'):
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return float(parsed[0]) if parsed else np.nan
                return float(parsed)
            return float(x)
    except (ValueError, SyntaxError):
        return np.nan
    return x

def clean_data_column(df: pd.DataFrame, col_name: str) -> np.ndarray:
    """
    Robustly parses a dataframe column into a numpy array of floats, 
    handling standard floats, strings, and stringified lists like '[0.83]'.
    """
    if col_name not in df.columns:
        return np.array([])
        
    raw_data = df[col_name].dropna().tolist()
    cleaned_data = []
    
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
            
    return np.array(cleaned_data)

def generate_qq_grids():
    """
    Generates a grid of Q-Q plots for each pipeline for all continuous metric or benchmark
    and displays the p-value in the subplot. Reads CSVs written by the stats pipeline
    (e.g. `faithfulness_scores.csv`) and the `normality_test_results.csv` file for
    annotated p-values.

    Saves each grid as `visualizations/qq_<eval_name>.png`.
    """
    normality_path = RESULTS_DIR / "normality_test_results.csv"
    normal_df = pd.read_csv(normality_path) if normality_path.exists() else pd.DataFrame()

    for eval in CONTINUOUS_EVALS:
        eval_name = eval.value[1]
        scores_path = RESULTS_DIR / f"{eval_name}_scores.csv"
        if not scores_path.exists():
            continue
        df = pd.read_csv(scores_path)

        pipelines = list(Pipeline)
        n = len(pipelines)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, pipeline in zip(axes, pipelines):
            col = pipeline.value[1] + "_score"
            if col not in df.columns:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue
            data = clean_data_column(df, col)
            if data.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue

            scipy_stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(pipeline.value[1])

            # annotate p-value from normality test if available
            if not normal_df.empty:
                row = normal_df.loc[(normal_df['eval_name'] == eval_name) & (normal_df['pipeline'] == pipeline.value[1])]
                if not row.empty:
                    p_val = row['p_value'].values[0]
                    if not pd.isna(p_val):
                        ax.text(0.05, 0.95, f"p={p_val:.3g}", transform=ax.transAxes, verticalalignment='top')

        fig.suptitle(f"Q-Q plots for {eval.value[0]}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = OUTPUT_DIR / f"qq_{eval_name}.png"
        fig.savefig(out_path)
        plt.close(fig)


def generate_box_plot_grids():
    """
    Generate one boxplot figure per continuous metric/benchmark and save to `visualizations/`.
    """
    for eval in CONTINUOUS_EVALS:
        eval_name = eval.value[1]
        scores_path = RESULTS_DIR / f"{eval_name}_scores.csv"
        if not scores_path.exists():
            continue
        df = pd.read_csv(scores_path)
        generate_box_plot(eval_name, df)


def generate_box_plot(eval_name: str, df: pd.DataFrame):
    """
    Create and save a boxplot comparing pipelines for `eval_name` using columns like 'tuned_score'.
    Adds a textbox with pipelines' mean rank (higher score => better rank).
    """
<<<<<<< HEAD
    score_cols = [c for c in df.columns if c.endswith("_score")]
    if not score_cols:
        return

    # Compute per-row ranks (higher score -> higher rank) and average
    ranks = df[score_cols].rank(axis=1, method='average', ascending=False)
    mean_ranks = ranks.mean(skipna=True)
    pipelines = [c.replace('_score', '') for c in score_cols]
    mean_ranks.index = pipelines

    melt_df = pd.melt(df, value_vars=score_cols, var_name='pipeline', value_name='score')
=======
    # Clean data before melting
    clean_df = df.copy()
    score_cols = [c for c in df.columns if c.endswith("_score")]
    for col in score_cols:
        # We need to assign back to the dataframe. 
        # clean_data_column returns np.ndarray.
        # We need to respect the index or just assign if lengths match.
        # However, clean_data_column drops NaNs. 
        # For boxplot with melt, we want to keep structure but just convert types.
        # If we drop NaNs, rows might misalign if we are not careful.
        # Actually, clean_data_column drops NaNs, which might be an issue if we just replace the column.
        # Better approach: Convert flexible, then drop NaNs specific to the plot if needed?
        # Or just parse robustly without dropping NaNs yet?
        
        # Let's write a localized parsing loop that preserves NaNs/structure for the dataframe
        # Or, just iterate and fix values.
        clean_df[col] = clean_df[col].apply(lambda x: parse_value_robust(x))

    melt_df = pd.melt(clean_df, id_vars=[c for c in df.columns if not c.endswith("_score")], value_vars=score_cols, var_name="pipeline", value_name="score")
    # normalize pipeline names to match Pipeline.value[1]
>>>>>>> 05c205073d980aca902a3c2efcf4793c25770ff6
    melt_df['pipeline'] = melt_df['pipeline'].str.replace('_score', '')

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x='pipeline', y='score', data=melt_df, order=pipelines)
    plt.title(f"{eval_name} — score distribution by pipeline")
    plt.xlabel("")
    plt.ylabel("Score")

    rank_lines = [f"{p}: {mean_ranks[p]:.3f}" for p in pipelines]
    rank_text = "\n".join(rank_lines)
    props = dict(boxstyle='round', facecolor='white', alpha=0.85)
    ax.text(0.98, 0.98, rank_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    out_path = OUTPUT_DIR / f"boxplot_{eval_name}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_bar_graph_grid():
    """
    Generate one bar graph per discrete benchmark and save to `visualizations/`.
    """
    for eval in DISCRETE_EVALS:
        eval_name = eval.value[1]
        scores_path = RESULTS_DIR / f"{eval_name}_scores.csv"
        if not scores_path.exists():
            continue
        df = pd.read_csv(scores_path)
        generate_bar_graph(eval_name, df)


def generate_bar_graph(eval_name: str, df: pd.DataFrame):
    """
    Create and save a bar graph comparing pipelines for `eval_name` using columns like '<pipeline>_score'.
    Computes proportion (mean) and standard error for binary/discrete scores and draws errorbars.
    """
    score_cols = [c for c in df.columns if c.endswith("_score")]
    if not score_cols:
        return

    stats = []
    for col in score_cols:
<<<<<<< HEAD
        values = df[col].dropna().to_numpy()
=======
        values = clean_data_column(df, col)
        if n is None:
            n = values.size
>>>>>>> 05c205073d980aca902a3c2efcf4793c25770ff6
        prop = np.nan
        se = np.nan
        if values.size > 0:
            prop = np.mean(values)
            se = math.sqrt(prop * (1 - prop) / max(1, values.size))
        pipeline_name = col.replace('_score', '')
        stats.append({'pipeline': pipeline_name, 'prop': prop, 'se': se})

    stats_df = pd.DataFrame(stats)
    order = stats_df['pipeline'].tolist()

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='pipeline', y='prop', data=stats_df, order=order, ci=None)
    ax.set_ylim(0, 1)
    plt.title(f"{eval_name} — proportion correct by pipeline")
    plt.ylabel('Proportion')
    plt.xlabel('')

    xcoords = np.arange(len(stats_df))
    props = stats_df['prop'].values
    ses = stats_df['se'].values
    ax.errorbar(xcoords, props, yerr=ses, fmt='none', ecolor='k', capsize=5)

    for i, val in enumerate(props):
        ax.text(i, val + 0.02, f"{val:.3f}", ha='center')

    out_path = OUTPUT_DIR / f"barplot_{eval_name}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_p_value_matrix(p_value_dict: dict, eval):
    """
    Build and save a symmetric p-value matrix heatmap from a mapping of pipeline-pair -> p-value.

    - Keys are tuples (`(p1, p2)`).
    - Self-comparisons are not expected in the input; the function sets diagonal entries to 1.0.
    - Missing comparisons remain NaN.

    Returns the square pandas DataFrame of p-values.
    """
    if p_value_dict is None:
        return None
    if not isinstance(p_value_dict, dict):
        raise TypeError("p_value_dict must be a dict mapping pipeline-pairs to p-values")

    if len(p_value_dict) == 0:
        return None

    # collect pipeline names
    pipelines = set()
    for key in p_value_dict.keys():
        if isinstance(key, (list, tuple)):
            p1, p2 = key
        else:
            parts = str(key).split('|')
            if len(parts) != 2:
                raise ValueError(f"Invalid key format: {key}")
            p1, p2 = parts
        pipelines.add(p1)
        pipelines.add(p2)

    pipelines = sorted(pipelines)
    piv = pd.DataFrame(index=pipelines, columns=pipelines, data=np.nan, dtype=float)

    # fill provided values symmetrically
    for key, p in p_value_dict.items():
        if isinstance(key, (list, tuple)):
            p1, p2 = key
        else:
            p1, p2 = str(key).split('|')
        piv.loc[p1, p2] = float(p)
        piv.loc[p2, p1] = float(p)

    # set diagonal to 1.0 (self-comparison)
    for p in pipelines:
        piv.loc[p, p] = 1.0

    piv = piv.reindex(index=pipelines, columns=pipelines)

    # build a colormap that has a sharp color transition at p=0.05
    # allocate more discrete colors on both sides but ensure a hard boundary at 0.05
    n_low = 50
    n_high = 206
    # gradient for significant (p <= 0.05) and non-significant (p > 0.05)
    low_colors = sns.color_palette("Reds", n_low)
    high_colors = sns.color_palette("Blues", n_high)
    colors = list(low_colors) + list(high_colors)
    cmap = ListedColormap(colors)

    # boundaries: create edges such that there's an exact boundary at 0.05
    low_bounds = np.linspace(0.0, 0.05, n_low, endpoint=False)
    high_bounds = np.linspace(0.05, 1.0, n_high + 1)
    boundaries = np.concatenate([low_bounds, high_bounds])
    norm = BoundaryNorm(boundaries, ncolors=len(colors))

    plt.figure(figsize=(7, 5))
    sns.heatmap(piv, annot=True, fmt='.3f', cmap=cmap, norm=norm,
                cbar_kws={'label': 'p-value'})
    plt.title(f'Heatmap of the P-Values of the Post-Hoc Tests for {eval.value[0]}')
    out_path = OUTPUT_DIR / f'p_value_matrix_{eval.value[1]}.png'
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return piv
