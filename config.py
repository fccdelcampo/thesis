from enum import Enum

class Pipeline(Enum):
    PROLOG_GRAPHRAG = "Prolog-GraphRAG Pipeline"
    GRAPHRAG = "GraphRAG Pipeline"
    TUNED = "Instruction-Tuned LLM Pipeline"
    
class Metric(Enum):
    FAITHFULNESS = "Faithfulness"
    ANSWER_RELEVANCE = "Answer Relevance"
    CONTEXT_RECALL = "Context Recall"
    
class Benchmark(Enum):
    MMLU = "MMLU"
    HOTPOTQA = "HotpotQA"
    TRUTHFULQA = "TruthfulQA" 
    
BINARY_EVALS = [Benchmark.MMLU, Benchmark.HOTPOTQA]
CONTINUOUS_EVALS = [Metric.FAITHFULNESS, Metric.ANSWER_RELEVANCE, Metric.CONTEXT_RECALL, Benchmark.TRUTHFULQA]
    
ALPHA = 0.05