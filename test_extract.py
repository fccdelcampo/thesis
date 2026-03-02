import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from llm_config import MODEL_NAME
from prolog_graphrag_pipeline.graphrag.kbpedia_retriever import KBPediaRetriever
from graphrag_pipeline.graphrag.graphrag_driver import DebugOllamaLLM

print("Initializing Real LLM from Baseline GraphRAG...")
retriever_llm = DebugOllamaLLM(
    model_name=MODEL_NAME,
    model_params={
        "keep_alive": "5m", 
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "format": "json"
        }
    }
)

retriever = KBPediaRetriever(driver=None, llm=retriever_llm)

query = """Which of the following parts of a plant cell has a function that is most similar to the function of an animal skeleton?
A. cell membrane
B. cell wall
C. chloroplast
D. nucleus"""

print("Starting extract_entities with baseline DebugOllamaLLM...")
entities = retriever.extract_entities(query)
print("Extracted:", entities)
print("DONE")
