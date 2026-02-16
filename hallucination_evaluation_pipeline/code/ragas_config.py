import os
from langchain_ollama import ChatOllama, OllamaEmbeddings

# RAGAS 0.4.x requires wrapped LLM/Embeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Define the model to use for both generation (evaluation) and embeddings
# Using the one verified to be available: llama3.1:8b-instruct-q4_K_M
MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
EMBED_MODEL_NAME = "nomic-embed-text"

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
