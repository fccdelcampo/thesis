import os
import sys

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings


# RAGAS 0.4.x requires wrapped LLM/Embeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Model name imported from the central config
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from llm_config import MODEL_NAME, RAGAS_MODEL_NAME
EMBED_MODEL_NAME = "nomic-embed-text"

# Add aliases for RAGAS
RAGAS_LLM_MODEL = RAGAS_MODEL_NAME
RAGAS_EMBEDDINGS_MODEL = EMBED_MODEL_NAME

def get_ragas_llm():
    """
    Returns a RAGAS-compatible wrapped LLM instance.
    """
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0, # Deterministic for evaluation
    )
    return LangchainLLMWrapper(llm)

def get_ragas_embeddings():
    """
    Returns a RAGAS-compatible wrapped Embeddings instance.
    """
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL_NAME
    )
    return LangchainEmbeddingsWrapper(embeddings)
