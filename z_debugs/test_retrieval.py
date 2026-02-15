
import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory of 'prolog_graphrag_pipeline' to sys.path
# And also 'prolog_graphrag_pipeline' itself so 'graphrag' can be found?
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "prolog_graphrag_pipeline"))

from neo4j import GraphDatabase
from prolog_graphrag_pipeline.graphrag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, RETRIEVER_OPTIONS, RETRIEVER, RETRIEVAL_QUERY
from prolog_graphrag_pipeline.graphrag.retriever import create_retriever
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.schema import SchemaBuilder
from neo4j_graphrag.types import LLMModel

# Schema for test
SCHEMA = {
    "node_types": ["Concept", "Person", "Event"],
    "relationship_types": ["RELATED_TO", "HAS_CAPITAL"]
}

async def test_pipeline():
    print("Initializing components...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    # Use Ollama Embeddings (Llama 3 8B = 4096 dim)
    model_name = "llama3.1:8b-instruct-q4_K_M" 
    embedder = OllamaEmbeddings(model=model_name)
    
    # Use Ollama for KG Builder
    kg_llm = OpenAILLM(
        model_name="llama3.1:8b-instruct-q4_K_M",
        model_params={"max_tokens": 1000},
        base_url="http://localhost:11434/v1",
        api_key="ollama" 
    )

    # 1. Ingest Data
    print("Ingesting test data...")
    
    # We use SimpleKGPipeline
    kg_builder = SimpleKGPipeline(
        driver=driver,
        llm=kg_llm,
        embedder=embedder,
        entities=SCHEMA["node_types"],
        relations=SCHEMA["relationship_types"],
        from_pdf=False
    )
    
    text = "France is a country in Europe. Its capital is Paris. Paris is a beautiful city."
    print(f"Processing text: '{text}'")
    
    # SimpleKGPipeline.run_async takes text
    await kg_builder.run_async(text=text)
    print("Ingestion complete.")

    # 2. Create Retriever (and Indexes)
    print("Creating Retriever and Indexes...")
    retriever = create_retriever(driver, embedder)
    if not retriever:
        print("Failed to update retriever.")
        return

    # 3. Search
    query = "What is the capital of France?"
    print(f"Searching for: '{query}'")
    
    results = retriever.search(query_text=query, top_k=3)
    
    print(f"Found {len(results.items)} chunks.")
    for item in results.items:
        print(f"Content: {item.content}")
        print(f"Metadata: {item.metadata}")
        
    driver.close()

if __name__ == "__main__":
    asyncio.run(test_pipeline())
