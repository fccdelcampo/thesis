from enum import Enum

ALPHA = 0.05
NUM_TEST_CASES = 100

class Pipeline(Enum):
    TUNED = (0, "tuned")
    GRAPHRAG = (1, "graphrag")
    PROLOG_GRAPHRAG = (2, "prolog_graphrag")

class Metric(Enum):
    FAITHFULNESS = (0, "faithfulness")
    ANSWER_RELEVANCE = (1, "answer_relevance")
    CONTEXT_RECALL = (2, "context_recall")

class Benchmark(Enum):
    MMLU = (0, "mmlu")
    TRUTHFULQA = (1, "truthfulqa")
    HOTPOTQA = (2, "hotpotqa")

CONTINUOUS_EVALS = [
    Metric.FAITHFULNESS,
    Metric.ANSWER_RELEVANCE,
    Metric.CONTEXT_RECALL,
    Benchmark.TRUTHFULQA
]

DISCRETE_EVALS = [
    Benchmark.MMLU,
    Benchmark.HOTPOTQA
]
