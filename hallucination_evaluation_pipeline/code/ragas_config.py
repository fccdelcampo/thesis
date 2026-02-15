import os
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Define the model to use for both generation (evaluation) and embeddings
# Using the one verified to be available: llama3.1:8b-instruct-q4_K_M
MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"

def get_ragas_llm():
    """
    Returns a LangChain ChatOllama instance configured for RAGAS.
    """
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0, # Deterministic for evaluation
    )

def get_ragas_embeddings():
    """
    Returns a LangChain OllamaEmbeddings instance configured for RAGAS.
    """
    return OllamaEmbeddings(
        model=MODEL_NAME
    )
