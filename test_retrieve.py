
import sys
import os
import asyncio

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prolog_graphrag_pipeline.graphrag.config import RETRIEVAL_QUERY, RETRIEVER, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from prolog_graphrag_pipeline.graphrag.graphrag_driver import initialize_models, create_retriever
from neo4j import GraphDatabase

def test_retrieval():
    print("Initializing models...")
    # initialize_models returns (encoder_llm, retriever_llm, embedder)
    _, _, embedder = initialize_models()
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    print("Creating retriever...")
    retriever = create_retriever(driver, embedder)
    
    query = "What comes after 1?"
    print(f"\nTesting Query: {query}")
    
    try:
        results = retriever.search(query_text=query, top_k=3)
        
        print(f"\nFound {len(results.items)} items.")
        
        for i, item in enumerate(results.items):
            print(f"\n--- Item {i+1} ---")
            import json
            print(f"Content: {item.content[:100]}...")
            print(f"Metadata: {json.dumps(item.metadata, indent=2)}")
            
    except Exception as e:
        print(f"Retrieval failed: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    test_retrieval()
