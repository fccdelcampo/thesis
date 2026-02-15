import os
import sys

# Add project root and stats directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
pipeline_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if pipeline_root not in sys.path:
    sys.path.append(pipeline_root)

import answer_relevance
import context_recall
import faithfulness
import hotpotqa
import mmlu
import truthfulqa
from stats import stats_driver

def run_hallucination_evaluation_pipeline():
    """
    Called to run all test cases for the hallucination evaluation  pipeline,
    then calls performStatisticalTests().
    """
    print("Running Answer Relevance...")
    answer_relevance.main()
    print("Running Context Recall...")
    context_recall.main()
    print("Running Faithfulness...")
    faithfulness.main()
    
    # These seem to be benchmark evaluators, also run them
    print("Running HotpotQA...")
    hotpotqa.main()
    print("Running MMLU...")
    mmlu.main()
    print("Running TruthfulQA...")
    truthfulqa.main()
    
    print("Running Statistical Tests...")
    stats_driver.perform_statistical_tests()

if __name__ == "__main__":
    run_hallucination_evaluation_pipeline()