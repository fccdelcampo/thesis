import sys
import csv
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
IO_DIR = ROOT / "explainability_evaluation_results"

def calculate_comprehensiveness(test_case_id: int, df: pd.DataFrame) -> float:
    se_x_c = df.loc[df["test_case_id"] == test_case_id, "se_x_c"].values[0]
    se_x = df.loc[df["test_case_id"] == test_case_id, "se_x"].values[0]
    score = se_x_c - se_x
    log_row(test_case_id=test_case_id, comprehensiveness=score)

def log_row(test_case_id, comprehensiveness) -> None:
    with open(IO_DIR / "comprehensiveness.csv", 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_case_id", "comprehensiveness"]
        )
        writer.writerow({
            "test_case_id": test_case_id,
            "comprehensiveness": comprehensiveness,
        })

def run_comprehensiveness_computations() -> None:
    df = pd.read_csv(IO_DIR / "semantic_entropy.csv")
    test_case_ids = df["test_case_id"].tolist()
    for id in test_case_ids:
        calculate_comprehensiveness(test_case_id=id, df=df)