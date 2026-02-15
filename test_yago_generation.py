
import sys
import os
import asyncio
import json

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prolog_graphrag_pipeline.graphrag.config import RETRIEVAL_QUERY, RETRIEVER, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from prolog_graphrag_pipeline.graphrag.graphrag_driver import initialize_models, create_retriever
from prolog_graphrag_pipeline.graphrag.retriever import generate
from neo4j import GraphDatabase

def test_generation():
    print("Initializing models...")
    # initialize_models returns (encoder_llm, retriever_llm, embedder)
    _, retriever_llm, embedder = initialize_models()
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    print("Creating retriever...")
    retriever = create_retriever(driver, embedder)
    
    query = "What is the capital of France?"
    # query = "When was the Golden Gate Bridge built in New York?"
    print(f"\nGenerative Test Query: {query}")
    
    try:
        # Calls the generate function in retriever.py which uses GRAPHRAG_TEMPLATE
        result = generate(llm=retriever_llm, retriever=retriever, query=query)
        
        print("\n--- LLM Response ---")
        print(result["answer"])
        
        print("\n--- Retrieved Context Summary ---")
        for ctx in result["contexts"]:
            print(ctx[:1000] + "...")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Generation failed: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    test_generation()
