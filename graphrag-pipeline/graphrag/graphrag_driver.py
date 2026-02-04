import neo4j
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
import os
import glob
from .config import PROMPT_TEMPLATE, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD # added dot for relative import, assuming config.py is in the same directory as graphrag_driver.py
import asyncio

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
# llama3.1:8b-instruct uses 8GB VRAM, while q4_K_M quantized version uses around 4GB-5GB VRAM so it's the only one I can run locally
MODEL = "llama3.1:8b-instruct-q4_K_M"

from llm import generate

neo4j_driver = neo4j.GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

def initialize_models() -> tuple[OllamaLLM, OllamaLLM, OllamaEmbeddings]:
    # keep_alive=0 unloads VRAM after each document processing
    encoder_llm = OllamaLLM(
        model_name=MODEL,
        model_params={
            "num_ctx": 4096,
            "temperature": 0,
            "format": "json",
            "keep_alive": 0
        }
    )

    # 2. The Retriever: Logical synthesis for Prolog
    retriever_llm = OllamaLLM(
        model_name=MODEL,
        model_params={
            "num_ctx": 2048, # Smaller window is faster for QA
            "temperature": 0.1, 
            "keep_alive": "5m" # Keep it loaded during an active chat session
        }
    )

    # 2. Initialize Embedder
    embedder = OllamaEmbeddings(
        model=MODEL
    )

    print("Ollama loaded successfully.")
    
    return encoder_llm, retriever_llm, embedder

def setup_kg_pipeline(llm, embedder) -> SimpleKGPipeline    :
    chunk_splitter = FixedSizeSplitter(chunk_size=2000, chunk_overlap=200)

    kg_builder_pdf = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=embedder,
        schema="EXTRACTED",
        prompt_template=PROMPT_TEMPLATE,
        from_pdf=True,
        text_splitter=chunk_splitter
    )
    
    kg_builder_text = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=embedder,
        schema="EXTRACTED",
        prompt_template=PROMPT_TEMPLATE,
        from_pdf=False,
        text_splitter=chunk_splitter
    )
    
    return kg_builder_pdf, kg_builder_text

def generate_answer(query, retriever, llm) -> dict:
    return generate(llm, retriever, query)

from .encoder import process_pdf_documents, process_text_context, extract_query_and_context

async def test_process_pdf_context(kg_builder_pdf: SimpleKGPipeline):
    await process_pdf_documents(kg_builder_pdf)
    
async def test_process_text_context(kg_builder_text: SimpleKGPipeline, query: str):
    await process_text_context(kg_builder_text)
    
def test_extract_query_and_context(user_prompt: str):
    query, context = extract_query_and_context(user_prompt)
    print(f"Extracted query: {query}")
    print(f"Extracted context: {context}")
    return query, context
    
async def perform_tests():
    # 1. Initialize models and pipelines
    llm, retriever_llm, embedder = initialize_models() # forgot retriever_llm here, but it's not used in the tests below, so it's fine for now
    kg_builder_pdf, kg_builder_text = setup_kg_pipeline(llm, embedder)
    
    # 2. Extract Phase (Synchronous)
    print("--- Testing extract_query_and_context ---")
    question = "Which of the following sets is defined by the Well-Ordering Principle as having a guaranteed smallest element? A) Empty sets B) Sets of negative integers C) Nonempty sets of nonnegative integers D) Sets of positive rationals"
    
    # Corrected: Removed the extra comma that was making this a tuple
    query, context = test_extract_query_and_context(question)

    # 3. Processing Phase (Asynchronous)
    # Corrected: Do NOT use 'await' INSIDE asyncio.gather. 
    # Pass the coroutines themselves to gather.
    print("\n--- Starting Knowledge Graph Ingestion ---")
    
    await asyncio.gather(
        test_process_pdf_context(kg_builder_pdf),
        process_text_context(kg_builder_text, context)
    )
    
    print("\n--- All tests completed ---")

def main():
    asyncio.run(perform_tests())

if __name__ == "__main__":
    main()
    
# def generate_context(query, retriever, llm) -> str:
#     return generate(llm, retriever, query)