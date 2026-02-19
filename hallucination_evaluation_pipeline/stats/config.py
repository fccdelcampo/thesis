from enum import Enum

ALPHA = 0.05
NUM_TEST_CASES = 100

class Pipeline(Enum):
    PROLOG_GRAPHRAG = ("Prolog-GraphRAG Pipeline", "prolog_graphrag")
    GRAPHRAG = ("GraphRAG Pipeline", "graphrag")
    TUNED = ("llama3.1:instruct", "tuned")

class Metric(Enum):
    FAITHFULNESS = ("Faithfulness", "faithfulness")
    ANSWER_RELEVANCY = ("Answer Relevancy", "answer_relevancy")
    CONTEXT_RECALL = ("Context Recall", "context_recall")
    
class Benchmark(Enum):
    MMLU = ("MMLU", "mmlu")
    HOTPOTQA = ("HotpotQA", "hotpotqa")
    TRUTHFULQA = ("TruthfulQA", "truthfulqa")

CONTINUOUS_EVALS = [
    Metric.FAITHFULNESS,
    Metric.ANSWER_RELEVANCY,
    Metric.CONTEXT_RECALL,
    Benchmark.TRUTHFULQA
]

DISCRETE_EVALS = [
    Benchmark.MMLU,
    Benchmark.HOTPOTQA
]
