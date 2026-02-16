import math
import os
import sys

# Optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import scipy.stats as scipy_stats
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import scikit_posthocs as sp
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    import matplotlib.patheffects as path_effects
    import matplotlib.patheffects as pe
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Visualization libraries missing: {e}. Visualization will be skipped.")
    VISUALIZATION_AVAILABLE = False
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
         print("Critical: Numpy/Pandas also missing.")

try:
    from .config import ALPHA, CONTINUOUS_EVALS, DISCRETE_EVALS, Pipeline
except ImportError:
    try:
        from config import ALPHA, CONTINUOUS_EVALS, DISCRETE_EVALS, Pipeline
    except ImportError:
        from stats.config import ALPHA, CONTINUOUS_EVALS, DISCRETE_EVALS, Pipeline

import os

# Prefer results folder at workspace root: ../../hallucination_evaluation_results/visualizations
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DATA_DIR = os.path.join(ROOT, "hallucination_evaluation_results", "data")  # Adjusted path
# stats_driver says IO_DIR = ROOT / "hallucination_evaluation_results".
# And csv files are there.
# But original code said: RESULTS_DATA_DIR = os.path.join(ROOT, "hallucination_evaluation_results", "data")
# visualizer_generator here expected "data" subdir.
# visualizer_generator expected "data" subdir.

RESULTS_DATA_DIR = os.path.join(ROOT, "hallucination_evaluation_results")
OUTPUT_DIR = os.path.join(ROOT, "hallucination_evaluation_results", "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    and displays the p-value in the subplot. First reads relevant CSV files in hallucination_evaluation_results/ 
    to load data and normality test p-values.
    Saves the final grid in ../visualizations/ relative to this file.

    """
    if not VISUALIZATION_AVAILABLE:
        print("Skipping generate_qq_grids: Visualization libraries not installed.")
        return
    # Read normality test results if available
    normality_path = os.path.join(RESULTS_DATA_DIR, "normality_test_results.csv")
    if os.path.exists(normality_path):
        normal_df = pd.read_csv(normality_path)
    else:
        normal_df = pd.DataFrame()

    for eval in CONTINUOUS_EVALS:
        eval_name = eval.value[1]
        scores_path = os.path.join(RESULTS_DATA_DIR, f"{eval_name}_scores.csv")
        if not os.path.exists(scores_path):
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
            # Create QQ plot
            scipy_stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(pipeline.value[1])
            # annotate p-value from normality test if available
            p_val = None
            if not normal_df.empty:
                row = normal_df.loc[(normal_df['eval_name'] == eval_name) & (normal_df['pipeline'] == pipeline.value[1])]
                if not row.empty:
                    p_val = row['p_value'].values[0]
            if p_val is not None and not (pd.isna(p_val)):
                ax.text(0.05, 0.95, f"p={p_val:.3g}", transform=ax.transAxes, verticalalignment='top')

        fig.suptitle(f"Q-Q plots for {eval.value[0]}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(OUTPUT_DIR, f"qq_{eval_name}.png")
        fig.savefig(out_path)
        plt.close(fig)

def generate_box_plot_grids():
    """
    Generate one grid of boxplots per continuous metric and benchmark. Each grid compares, for each metric,
    the distributions across the three pipelines. Will read relevant files in hallucination_evaluation_results/

    """
    for eval in CONTINUOUS_EVALS:
        eval_name = eval.value[1]
        scores_path = os.path.join(RESULTS_DATA_DIR, f"{eval_name}_scores.csv")
        if not os.path.exists(scores_path):
            continue
        df = pd.read_csv(scores_path)
        generate_box_plot(eval_name, df)

def generate_box_plot(eval_name: str, df: pd.DataFrame):
    """
    Create and save a boxplot comparing pipelines for `eval_name` using columns like 'tuned_score'.
    """
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
    melt_df['pipeline'] = melt_df['pipeline'].str.replace('_score', '')

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='pipeline', y='score', data=melt_df)
    plt.title(f"{eval_name} — score distribution by pipeline")
    plt.xlabel("")
    plt.ylabel("Score")
    out_path = os.path.join(OUTPUT_DIR, f"boxplot_{eval_name}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
def generate_bar_graph_grid():
    """
    Generate one grid of bar graphs per discrete metric and benchmark. Each grid compares, for each metric,
    the number of 0 or 1 labels across the three pipelines. Will read relevant files in hallucination_evaluation_results/

    """
    # Use discrete evals (e.g., MMLU, HotpotQA)
    for eval in DISCRETE_EVALS:
        eval_name = eval.value[1]
        scores_path = os.path.join(RESULTS_DATA_DIR, f"{eval_name}_scores.csv")
        if not os.path.exists(scores_path):
            continue
        df = pd.read_csv(scores_path)
        generate_bar_graph(eval_name, df)
    

def generate_bar_graph(eval_name: str, df: pd.DataFrame):
    """
    Create and save a bar graph comparing pipelines for `eval_name` using columns like 'tuned_score'.
    """
    # select score columns
    score_cols = [c for c in df.columns if c.endswith("_score")]
    if len(score_cols) == 0:
        return

    # compute proportions and standard error for binary/discrete scores
    stats = []
    n = None
    for col in score_cols:
        values = clean_data_column(df, col)
        if n is None:
            n = values.size
        prop = np.nan
        se = np.nan
        if values.size > 0:
            prop = np.mean(values)
            se = math.sqrt(prop * (1 - prop) / max(1, values.size))
        pipeline_name = col.replace('_score', '')
        stats.append({'pipeline': pipeline_name, 'prop': prop, 'se': se})

    stats_df = pd.DataFrame(stats)

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='pipeline', y='prop', data=stats_df, yerr=stats_df['se'])
    ax.set_ylim(0, 1)
    plt.title(f"{eval_name} — proportion correct by pipeline")
    plt.ylabel('Proportion')
    plt.xlabel('')

    # annotate values
    for i, row in stats_df.iterrows():
        ax.text(i, row['prop'] + 0.02, f"{row['prop']:.3f}", ha='center')

    out_path = os.path.join(OUTPUT_DIR, f"barplot_{eval_name}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
   

def generate_p_value_matrix(df: pd.DataFrame):
    """
    Will accept a DataFrame consisting of pairs of pipelines and their posthoc test p-value to construct a heatmap of p-values
    """
    # Accept rows of pairwise comparisons in either (loser,winner,p_value) or (pipeline1,pipeline2,pvalue)
    if df.shape[0] == 0:
        return None

    # Normalize possible column names
    if set(["pipeline1", "pipeline2", "pvalue"]).issubset(df.columns):
        rows = df[['pipeline1', 'pipeline2', 'pvalue']].rename(columns={'pipeline1': 'p1', 'pipeline2': 'p2', 'pvalue': 'p'})
    elif set(["loser", "winner", "p_value"]).issubset(df.columns):
        rows = df[['loser', 'winner', 'p_value']].rename(columns={'loser': 'p1', 'winner': 'p2', 'p_value': 'p'})
    else:
        # Assume square matrix given
        piv = df.copy()
        piv = piv.astype(float)
        np.fill_diagonal(piv.values, 0)
        plt.figure(figsize=(6, 5))
        sns.heatmap(piv, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'p-value'})
        plt.title('Post-hoc pairwise p-values')
        out_path = os.path.join(OUTPUT_DIR, 'p_value_matrix.png')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return piv

    # Build square matrix from rows
    pipelines = sorted(pd.unique(rows[['p1', 'p2']].values.ravel('K')))
    piv = pd.DataFrame(index=pipelines, columns=pipelines, data=np.nan, dtype=float)
    for _, r in rows.iterrows():
        p1 = r['p1']
        p2 = r['p2']
        p = float(r['p'])
        piv.loc[p1, p2] = p
        piv.loc[p2, p1] = p

    for c in piv.columns:
        if c not in piv.index:
            piv.loc[c] = np.nan
    piv = piv.reindex(index=piv.columns)
    np.fill_diagonal(piv.values, 0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(piv, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'p-value'})
    plt.title('Post-hoc pairwise p-values')
    out_path = os.path.join(OUTPUT_DIR, 'p_value_matrix.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return piv
    