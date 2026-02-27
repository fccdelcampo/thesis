from enum import Enum

ALPHA = 0.05
NUM_TEST_CASES = 100

class Pipeline(Enum):
    PROLOG_GRAPHRAG = ("Prolog-GraphRAG Pipeline", "prolog_graphrag")
    GRAPHRAG = ("GraphRAG Pipeline", "graphrag")
    TUNED = ("llama3.1:8b-instruct-q4_K_M", "tuned")

class Metric(Enum):
    FAITHFULNESS = ("Faithfulness", "faithfulness")
    ANSWER_RELEVANCY = ("Answer Relevancy", "answer_relevancy")
    CONTEXT_RECALL = ("Context Recall", "context_recall")
    
class Benchmark(Enum):
    MMLU = ("MMLU", "mmlu")
    AI2_ARC = ("AI2 ARC", "ai2_arc")
    OPENBOOKQA = ("OpenBookQA", "openbookqa")

CONTINUOUS_EVALS = [
    Metric.FAITHFULNESS,
    Metric.ANSWER_RELEVANCY,
    Metric.CONTEXT_RECALL,
]

DISCRETE_EVALS = [
    Benchmark.MMLU,
    Benchmark.AI2_ARC,
    Benchmark.OPENBOOKQA,
]
