
import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hallucination_evaluation_pipeline.stats import stats_driver

print("Running stats only...")
stats_driver.perform_statistical_tests()
print("Stats completed.")
