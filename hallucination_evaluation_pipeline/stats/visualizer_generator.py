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
            ax.set_title(pipeline.value[0])

            # annotate p-value from normality test if available
            if not normal_df.empty:
                row = normal_df.loc[(normal_df['eval_name'] == eval_name) & (normal_df['pipeline'] == pipeline.value[1])]
                if not row.empty:
                    p_val = row['p_value'].values[0]
                    # display p-value and interpret normality under it
                    if not pd.isna(p_val):
                        try:
                            p_float = float(p_val)
                        except Exception:
                            p_float = np.nan
                        ax.text(0.05, 0.95, f"p={p_float:.3g}", transform=ax.transAxes, verticalalignment='top')
                        interp = 'NON-NORMAL' if (not pd.isna(p_float) and p_float < ALPHA) else 'NORMAL'
                        props = dict(boxstyle='round', facecolor='white', alpha=0.85)
                        ax.text(0.05, 0.88, interp, transform=ax.transAxes, verticalalignment='top', bbox=props)

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
        generate_box_plot(eval, df)

def file_to_plain(pipeline_name: str):
    if pipeline_name in ["prolog_graphrag_score", "prolog_graphrag"]:
        return "Prolog-GraphRAG"
    if pipeline_name in ["graphrag_score", "graphrag"]:
        return "GraphRAG"
    if pipeline_name in ["tuned_score", "tuned"]:
        return "LLaMa 3.1 Instruct"
    raise ValueError(f"Invalid pipeline name: {pipeline_name}")

def generate_box_plot(eval: str, df: pd.DataFrame):
    """
    Create and save a boxplot comparing pipelines for `eval_name` using columns like 'tuned_score'.
    Adds a textbox with pipelines' mean rank (higher score => better rank).
    """
    score_cols = [c for c in df.columns if c.endswith("_score")]
    if not score_cols:
        return
    
    print("SCORE COLS: ", score_cols)
    print(f"{eval.value[0]} BEFORE MELT DF: ")
    print(df[score_cols])
    print()

    # Compute per-row ranks (higher score -> higher rank) and average
    ranks = df[score_cols].rank(axis=1, method='average')
    
    mean_ranks = ranks.mean(skipna=True)
    pipelines = [c for c in score_cols]
    mean_ranks.index = pipelines

    melt_df = pd.melt(df, value_vars=score_cols, var_name='pipeline', value_name='score')
    melt_df['pipeline'] = melt_df['pipeline'].str.replace('_score', '').replace("tuned", "LLaMa 3.1 Instruct").replace("prolog_graphrag", "Prolog-GraphRAG").replace("graphrag", "GraphRAG")
    
    print(f"{eval.value[0]} MELT DF: ")
    print(melt_df)
    print()
    plt.figure(figsize=(8, 6))
    print('pipelines variable:', pipelines)
    print('unique pipelines in melt_df:', melt_df['pipeline'].unique())
    print('counts:', melt_df['pipeline'].value_counts().to_dict())
    ax = sns.boxplot(x='pipeline', y='score', data=melt_df)
    plt.title(f"{eval.value[0]} — score distribution by pipeline")
    plt.xlabel("")
    plt.ylabel("Score")

    rank_lines = [f"{file_to_plain(p)} Mean Rank: {mean_ranks[p]:.3f}" for p in pipelines]
    rank_text = "\n".join(rank_lines)
    props = dict(boxstyle='round', facecolor='white', alpha=0.85)
    ax.text(0.98, 0.98, rank_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # add hypothesis test p-value and interpretation textbox (if available)
    hypothesis_path = RESULTS_DIR / "hypothesis_test_results.csv"
    hyp_text = "p=NA\nUNKNOWN"
    if hypothesis_path.exists():
        try:
            hyp_df = pd.read_csv(hypothesis_path)
            row = hyp_df.loc[hyp_df['eval_name'] == eval.value[1]]
            if not row.empty:
                raw_p = row['p_value'].values[0]
                p_val = parse_value_robust(raw_p)
                interp = row['interpretation'].values[0] if 'interpretation' in row.columns else (
                    'SIGNIFICANT' if (not pd.isna(p_val) and float(p_val) < ALPHA) else 'NON-SIGNIFICANT'
                )
                if pd.isna(p_val):
                    hyp_text = "p=NA\nERROR"
                else:
                    hyp_text = f"p={float(p_val):.3g}\n{interp}"
        except Exception:
            hyp_text = "p=NA\nERROR"

    ax.text(0.02, 0.98, hyp_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', bbox=props)

    out_path = OUTPUT_DIR / f"boxplot_{eval.value[1]}.png"
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
        generate_bar_graph(eval, df)


def generate_bar_graph(eval, df: pd.DataFrame):
    """
    Create and save a bar graph comparing pipelines for `eval_name` using columns like '<pipeline>_score'.
    Computes proportion (mean) and standard error for binary/discrete scores and draws errorbars.
    """
    score_cols = [c for c in df.columns if c.endswith("_score")]
    if not score_cols:
        return

    stats = []
    for col in score_cols:
        values = df[col].dropna().to_numpy()
        prop = np.nan
        se = np.nan
        if values.size > 0:
            prop = np.mean(values)
            se = math.sqrt(prop * (1 - prop) / max(1, values.size))
        pipeline_name = col.replace('_score', '')
        stats.append({'pipeline': pipeline_name, 'prop': prop, 'se': se})

    stats_df = pd.DataFrame(stats)
    stats_df['pipeline'] = stats_df['pipeline'].str.replace('_score', '').replace("tuned", "LLaMa 3.1 Instruct").replace("prolog_graphrag", "Prolog-GraphRAG").replace("graphrag", "GraphRAG")

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='pipeline', y='prop', data=stats_df, ci=None)
    ax.set_ylim(0, 1)
    plt.title(f"{eval.value[0]} — proportion correct by pipeline")
    plt.ylabel('Proportion')
    plt.xlabel('')

    xcoords = np.arange(len(stats_df))
    props = stats_df['prop'].values
    ses = stats_df['se'].values
    ax.errorbar(xcoords, props, yerr=ses, fmt='none', ecolor='k', capsize=5)

    # place the mean value centered inside each bar with larger font
    for i, val in enumerate(props):
        # y position at middle of the bar; ensure a small offset for extremely small bars
        y = val / 2.0
        y = y if y > 0.02 else 0.02
        ax.text(i, y, f"{val:.3f}", ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # add hypothesis test p-value and interpretation textbox (if available)
    hypothesis_path = RESULTS_DIR / "hypothesis_test_results.csv"
    hyp_text = "p=NA\nUNKNOWN"
    if hypothesis_path.exists():
        try:
            hyp_df = pd.read_csv(hypothesis_path)
            row = hyp_df.loc[hyp_df['eval_name'] == eval.value[1]]
            if not row.empty:
                raw_p = row['p_value'].values[0]
                p_val = parse_value_robust(raw_p)
                interp = row['interpretation'].values[0] if 'interpretation' in row.columns else (
                    'SIGNIFICANT' if (not pd.isna(p_val) and float(p_val) < ALPHA) else 'NON-SIGNIFICANT'
                )
                if pd.isna(p_val):
                    hyp_text = "p=NA\nERROR"
                else:
                    hyp_text = f"p={float(p_val):.3g}\n{interp}"
        except Exception:
            hyp_text = "p=NA\nERROR"

    props = dict(boxstyle='round', facecolor='white', alpha=0.85)
    ax.text(0.02, 0.95, hyp_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', bbox=props)

    out_path = OUTPUT_DIR / f"barplot_{eval.value[1]}.png"
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
