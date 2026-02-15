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
    
class SampleSequencesFlags(Enum):
    X = "x"
    Q = "q"
    X_C = r"x\c"
    
DISCRETE_EVALS = [Benchmark.MMLU, Benchmark.HOTPOTQA]
CONTINUOUS_EVALS = [Metric.FAITHFULNESS, Metric.ANSWER_RELEVANCE, Metric.CONTEXT_RECALL, Benchmark.TRUTHFULQA]
    
ALPHA = 0.05
NUM_TEST_CASES = 100
ENTAILMENT_THRESHOLD = 0.7
CORRECTNESS_THRESHOLD = 0.7

CLEAR_DATA_PASSWORD = "hastalavista"