from enum import Enum

class Pipeline(Enum):
    PROLOG_GRAPHRAG = ("Prolog-GraphRAG Pipeline", "prolog_graphrag")
    GRAPHRAG = ("GraphRAG Pipeline", "graphrag")
    TUNED = ("llama3.1:instruct", "tuned")
    
for i, pipeline1 in enumerate(list(Pipeline)):
    print(f"pipeline: {pipeline1}")
    print(f"pipeline.value: {pipeline1.value}")
    print(f"p.value[0]: {pipeline1.value[0]}")
    print(f"p.value[1]: {pipeline1.value[1]}\n")
    
class Metric(Enum):
    FAITHFULNESS = ("Faithfulness", "faithfulness")
    ANSWER_RELEVANCY = ("Answer Relevancy", "answer_relevancy")
    CONTEXT_RECALL = ("Context Recall", "context_recall")
    
class Benchmark(Enum):
    MMLU = ("MMLU", "mmlu")
    HOTPOTQA = ("HotpotQA", "hotpotqa")
    TRUTHFULQA = ("TruthfulQA", "truthfulqa")
    
class SampleSequencesFlags(Enum):
    X = ("x", "x")
    Q = ("q", "q")
    X_C = ("x\\c", "x_c")
    
DISCRETE_EVALS = [Benchmark.MMLU, Benchmark.HOTPOTQA]
CONTINUOUS_EVALS = [Metric.FAITHFULNESS, Metric.ANSWER_RELEVANCY, Metric.CONTEXT_RECALL, Benchmark.TRUTHFULQA]
    
ALPHA = 0.05
NUM_TEST_CASES = 100
ENTAILMENT_THRESHOLD = 0.7
CORRECTNESS_THRESHOLD = 0.7

CLEAR_DATA_PASSWORD = "hastalavista"