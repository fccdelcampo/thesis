import sys
import csv
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
IO_DIR = ROOT / "explainability-evaluation-results"

def calculate_sufficiency(test_case_id: int, df: pd.DataFrame) -> float:
    se_q = df.loc[df["test_case_id"] == test_case_id, "se_q"].values[0]
    se_x = df.loc[df["test_case_id"] == test_case_id, "se_x"].values[0]
    score = se_q - se_x
    log_row(test_case_id=id, sufficiency=score)

def log_row(test_case_id, sufficiency) -> None:
    with open(IO_DIR / "sufficiency.csv", 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_case_id", "sufficiency"]
        )
        writer.writerow({
            "test_case_id": test_case_id,
            "sufficiency": sufficiency,
        })

def run_sufficiency_computations() -> None:
    df = pd.read_csv(IO_DIR / "semantic_entropy.csv")
    test_case_ids = df["test_case_id"].tolist()
    for id in test_case_ids:
        calculate_sufficiency(test_case_id=id, df=df)