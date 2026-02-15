
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(r"c:\Users\John Reniel\thesis\thesis")
sys.path.append(str(ROOT))

try:
    import explainability_evaluation_pipeline.driver
    print("Imported driver")
    import explainability_evaluation_pipeline.semantic_entropy
    print("Imported semantic_entropy")
    import explainability_evaluation_pipeline.comprehensiveness
    print("Imported comprehensiveness")
    import explainability_evaluation_pipeline.sufficiency
    print("Imported sufficiency")
    import explainability_evaluation_pipeline.auroc
    print("Imported auroc")
    print("All imports successful")
except Exception as e:
    print(f"Import failed: {e}")
