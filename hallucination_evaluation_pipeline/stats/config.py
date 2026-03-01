from enum import Enum
import sys, os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from llm_config import MODEL_NAME as _LLM_MODEL

ALPHA = 0.05
NUM_TEST_CASES = 100

class Pipeline(Enum):
    PROLOG_GRAPHRAG = ("Prolog-GraphRAG Pipeline", "prolog_graphrag")
    GRAPHRAG = ("GraphRAG Pipeline", "graphrag")
    TUNED = (_LLM_MODEL, "tuned")

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
