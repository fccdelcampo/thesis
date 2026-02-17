
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
    base_retriever = create_retriever(driver, embedder)
    from prolog_graphrag_pipeline.graphrag.retriever import FallbackRetriever
    retriever = FallbackRetriever(base_retriever)
    
    query = "When was the Golden Gate Bridge built in New York?"
    print(f"\nTesting Query: {query}")
    
    try:
        results = retriever.search(query_text=query, top_k=3)
        
        print(f"\nFound {len(results.items)} items.")
        
        import json
        with open("test_retrieve_output.txt", "w", encoding="utf-8") as f:
            for i, item in enumerate(results.items):
                f.write(f"\n--- Item {i+1} ---\n")
                f.write(f"Content: {item.content[:200]}...\n")
                f.write(f"Metadata: {json.dumps(item.metadata, indent=2)}\n")
        print("Done. Wrote results to test_retrieve_output.txt")
            
    except Exception as e:
        print(f"Retrieval failed: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    test_retrieval()
