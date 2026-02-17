import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
IO_DIR = ROOT / "explainability_evaluation_results"

import comprehensiveness, auroc, sufficiency, semantic_entropy
import sys, os
import csv
import random
import pandas as pd
from pathlib import Path
from config import DATA_CLEARANCE_PASSWORD
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
IO_DIR = ROOT / "explainability_evaluation_results"

def run_pipeline():
    semantic_entropy.run_semantic_entropy_computations()
    auroc.run_auroc_computations()
    comprehensiveness.run_comprehensiveness_computations()
    sufficiency.run_sufficiency_computations()
    
def interpret_results():
    auroc_df = pd.read_csv(IO_DIR / "auroc.csv")
    sufficiency_df = pd.read_csv(IO_DIR / "sufficiency.csv")
    comprehensiveness_df = pd.read_csv(IO_DIR / "comprehensiveness.csv")
    
    result_string = str()
    
    result_string += "============AUROC INTERPRETATION============"
    se_x_auroc = auroc_df["se_x"].values[0]
    se_q_auroc = auroc_df["se_q"].values[0]
    se_x_c_auroc = auroc_df["se_x_c"].values[0]
    result_string += fr"""
SE(x) AUROC Score: {se_x_auroc}
SE(x\\c) AUROC Score: {se_x_c_auroc}
SE(q) AUROC Score: {se_q_auroc}

    """
    
    result_string += "============COMPREHENSIVENESS INTERPRETATION============"
    comprehensiveness_values = comprehensiveness_df["comprehensiveness"].tolist()
    average_comprehensiveness = mean(comprehensiveness_values)
    result_string += fr"""
Average Comprehensiveness Score: {average_comprehensiveness}

    """
    
    result_string += "============SUFFICIENCY INTERPRETATION============"
    sufficiency_values = sufficiency_df["sufficiency"].tolist()
    average_sufficiency = mean(sufficiency_values)
    result_string += fr"""
Average Sufficiency Score: {average_sufficiency}

    """

def generate_artificial_data():
    with open(IO_DIR / "semantic_entropy.csv", "a", newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_case_id", "se_q", "se_q_label", "se_x", "se_x_label", "se_x_c", "se_x_c_label"]
        )
        for i in range(100):
            writer.writerow({
                "test_case_id": i+1,
                "se_q": random.uniform(0.1, 0.3),
                "se_q_label": random.randint(0, 1),
                "se_x": random.uniform(0.4, 0.6),
                "se_x_label": random.randint(0, 1),
                "se_x_c": random.uniform(0.7, 0.9),
                "se_x_c_label": random.randint(0, 1)
            })
            
def clear_datasets():
    password = input("You're about to clear all datasets under explainability_evaluation_results. Enter data clearance password -> ")
    if password == DATA_CLEARANCE_PASSWORD:
        headers = {
            "auroc.csv": "se_x,se_x_c,se_q\n",
            "comprehensiveness.csv": "test_case_id,comprehensiveness\n",
            "sufficiency.csv": "test_case_id,sufficiency\n",
            "semantic_entropy.csv": "test_case_id,se_q,se_q_label,se_x,se_x_label,se_x_c,se_x_c_label\n",
        }
        
        for filename, header in headers.items():
            path = os.path.join(IO_DIR, filename)
            with open(path, 'w', encoding='utf-8', newline='') as f:
                pass
    else:
        print("Wrong password")
        sys.exit()
        
def ensure_csv_headers():
    """Ensure all stats CSV files have their proper headers before writing data."""
    headers = {
            "auroc.csv": "se_x,se_x_c,se_q\n",
            "comprehensiveness.csv": "test_case_id,comprehensiveness\n",
            "sufficiency.csv": "test_case_id,sufficiency\n",
            "semantic_entropy.csv": "test_case_id,se_q,se_q_label,se_x,se_x_label,se_x_c,se_x_c_label\n",
        }
    
    for filename, header in headers.items():
        path = os.path.join(IO_DIR, filename)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(header)
            
if __name__ == "__main__":
    clear_datasets()
    ensure_csv_headers()
    generate_artificial_data()
    run_pipeline()
