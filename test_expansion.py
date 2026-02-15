import os
import sys

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prolog_graphrag_pipeline.graphrag.graphrag_driver import initialize_models, neo4j_driver
from prolog_graphrag_pipeline.graphrag.retriever import create_retriever, FallbackRetriever

def test_expansion_relevance():
    print("Initializing...")
    _, retriever_llm, embedder = initialize_models()
    base_retriever = create_retriever(neo4j_driver, embedder)
    
    # Pass LLM to enable expansion
    retriever = FallbackRetriever(base_retriever, llm=retriever_llm)
    
    query = "What comes after 1?"
    print(f"\n--- Testing Query: {query} ---")
    
    # Check expansion logic directly
    expanded = retriever.expand_query(query)
    print(f"\nExpanded Query: '{expanded}'")
    
    # Check retrieval results
    print("\nExecuting Retrieval...")
    results = retriever.search(query, top_k=5)
    
    print("\n--- Retrieved Items ---")
    for item in results.items:
        score = item.metadata.get('score', 0.0)
        print(f"[{score:.4f}] {item.content[:100]}...")

if __name__ == "__main__":
    test_expansion_relevance()
