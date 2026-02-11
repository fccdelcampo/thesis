from enum import Enum

class Pipeline(Enum):
    PROLOG_GRAPHRAG = ("Prolog-GraphRAG Pipeline", "prolog_graphrag")
    GRAPHRAG = ("GraphRAG Pipeline", "graphrag")
    TUNED = ("Instruction-Tuned LLM Pipeline", "tuned")
    
class Metric(Enum):
    FAITHFULNESS = ("Faithfulness", "faithfulness")
    ANSWER_RELEVANCE = ("Answer Relevance", "answer_relevance")
    CONTEXT_RECALL = ("Context Recall", "context_recall")
    
class Benchmark(Enum):
    MMLU = ("MMLU", "mmlu")
    HOTPOTQA = ("HotpotQA", "hotpotqa")
    TRUTHFULQA = ("TruthfulQA", "truthfulqa")
    
DISCRETE_EVALS = [Benchmark.MMLU, Benchmark.HOTPOTQA]
CONTINUOUS_EVALS = [Metric.FAITHFULNESS, Metric.ANSWER_RELEVANCE, Metric.CONTEXT_RECALL, Benchmark.TRUTHFULQA]
    
ALPHA = 0.05
NUM_TEST_CASES = 100