from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import random
import csv
import sys
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
IO_DIR = ROOT / "explainability_evaluation_results"

def compute_auroc_score(true_labels: dict, semantic_entropies: dict) -> None:
    print(F"\nTRUE LABELS\n{true_labels["SE(x)"]}\nSIZE: {len(true_labels["SE(x)"])}")
    print(F"\nSEMANTIC ENTROPIES\n{semantic_entropies["SE(x)"]}\nSIZE: {len(semantic_entropies["SE(x)"])}")
    se_x_auroc = roc_auc_score(y_true=true_labels["SE(x)"], y_score=semantic_entropies["SE(x)"])
    se_x_c_auroc = roc_auc_score(y_true=true_labels["SE(x\\c)"], y_score=semantic_entropies["SE(x\\c)"])
    se_q_auroc = roc_auc_score(y_true=true_labels["SE(q)"], y_score=semantic_entropies["SE(q)"])
    generate_roc_graph(true_labels=true_labels, semantic_entropies=semantic_entropies)
    se_results = {"SE(x)": se_x_auroc, "SE(x\\c)": se_x_c_auroc, "SE(q)": se_q_auroc}
    log_results(se_results)
    
def log_results(se_results: dict) -> None:
    with open(IO_DIR / "auroc.csv", 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["se_x", "se_x_c", "se_q"]
        )
        writer.writerow({
            "se_x": se_results["SE(x)"],
            "se_x_c": se_results["SE(x\\c)"],
            "se_q": se_results["SE(q)"]
        })

def generate_roc_graph(true_labels: dict, semantic_entropies: dict) -> None:
    """
    semantic_entropies: Dictionary containing {'SE(x)': [values], 'SE(q)': [values], ...}
    true_labels: Dictionary containing binary lists where 1 = Incorrect/Hallucination and 0 = Correct.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Standard color palette for different uncertainty states
    colors = {'SE(x)': 'darkblue', 'SE(q)': 'tab:red', 'SE(x\\c)': 'tab:orange'}
    # Light versions of colors for the shading/fill
    fill_colors = {'SE(x)': 'skyblue', 'SE(q)': 'lightcoral', 'SE(x\\c)': 'moccasin'}
    
    # 1. Plot each ROC curve and SHADE the area
    for name, scores in semantic_entropies.items():
        # Ensure we are predicting failures (label=1)
        fpr, tpr, _ = roc_curve(true_labels[name], scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot the main line
        ax.plot(fpr, tpr, label=f"{name} (AUROC = {roc_auc:.3f})", 
                color=colors.get(name, 'black'), lw=2.5)
        
        # FILL the area under this curve
        # alpha=0.2 prevents overlapping areas from becoming too dark
        ax.fill_between(
            fpr, tpr, 
            color=fill_colors.get(name, colors.get(name, 'gray')), 
            alpha=0.2
        )
        
    # 2. The Broken Line (Random Baseline)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guesser (AUROC = 0.5)")

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    
    # 3. TITLE & SUMMARY BOX (Centered Top)
    ax.set_title("Areas under the Receiver Operating Characteristic Curves (AUROC) for Semantic Entropy", pad=80, fontsize=14)
    
    # Dynamic string for the text box
    summary_stats = "\n".join([f"{n}: {auc(roc_curve(true_labels[n], s)[0], roc_curve(true_labels[n], s)[1]):.3f}" 
                               for n, s in semantic_entropies.items()])
    
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.5, 1.13, f"Validation Scores:\n{summary_stats}", transform=ax.transAxes, 
            fontsize=11, fontweight='bold', ha='center', va='top', bbox=props)

    # 4. THE LEGEND: Positioned below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
              fancybox=True, shadow=True, ncol=1)

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig("auroc.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def run_auroc_computations():
    df = pd.read_csv(IO_DIR / "semantic_entropy.csv")
    true_labels = {
        'SE(x)': df["se_x_label"].tolist(),
        'SE(x\\c)': df["se_x_c_label"].tolist(),
        'SE(q)': df["se_q_label"].tolist()
    }
    
    print(true_labels)
    
    semantic_entropies = {
        'SE(x)': df["se_x"].tolist(),
        'SE(x\\c)': df["se_x_c"].tolist(),
        'SE(q)': df["se_q"].tolist()
    }
    
    print(semantic_entropies)
    
    compute_auroc_score(true_labels=true_labels, semantic_entropies=semantic_entropies)
    
    
true_labels = {
    'SE(x)': [random.randint(0, 1) for i in range(100)],
    'SE(x\\c)': [random.randint(0, 1) for i in range(100)],
    'SE(q)': [random.randint(0, 1) for i in range(100)]
    }
semantic_entropies = {
    'SE(x)': [random.uniform(0.1, 0.3) for i in range(100)],
    'SE(q)': [random.uniform(0.4, 0.6) for i in range(100)],
    'SE(x\\c)': [random.uniform(0.7, 0.9) for i in range(100)],
    }
# print(compute_auroc_score(true_labels=true_labels, semantic_entropies=semantic_entropies))


