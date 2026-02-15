import answer_relevance, context_recall, faithfulness, hotpotqa, mmlu, truthfulqa
from stats import stats_driver

def run_hallucination_evaluation_pipeline():
    """
    Called to run all test cases for the hallucination evaluation  pipeline,
    then calls performStatisticalTests().
    """
    answer_relevance.main()
    context_recall.main()
    faithfulness.main()
    hotpotqa.main()
    mmlu.main()
    truthfulqa.main()
    stats_driver.performStatisticalTests