import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
IO_DIR = ROOT / "explainability_evaluation_results"

import comprehensiveness, auroc, sufficiency, semantic_entropy
import csv
import random

def load_test_cases() -> list[int]:
    pass

def run_pipeline():
    test_case_ids = load_test_cases()
    semantic_entropy.run_semantic_entropy_computations(test_case_ids)
    auroc.run_auroc_computations()
    comprehensiveness.run_comprehensiveness_computations()
    sufficiency.run_sufficiency_computations()

def generate_artificial_data():
    with open(IO_DIR / "semantic_entropy.csv", 'a', newline='') as f:
        for i in range(100):
            writer = csv.DictWriter(
                f,
                fieldnames=["test_case_id", "se_q", "se_x", "se_x_c"]
            )
            writer.writerow({
                "test_case_id": i+1,
                "se_q": random.randint(1, 10),
                "se_x": random.randint(1, 10),
                "se_x_c": random.randint(1, 10)
            })

def informal_tests():
    generate_artificial_data()
    comprehensiveness.run_comprehensiveness_computations()
    sufficiency.run_sufficiency_computations()


if __name__ == "__main__":
    informal_tests()
