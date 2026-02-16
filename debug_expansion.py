import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from prolog_graphrag_pipeline.graphrag.graphrag_driver import initialize_models, neo4j_driver
from prolog_graphrag_pipeline.graphrag.retriever import FallbackRetriever
from prolog_graphrag_pipeline.graphrag.config import RETRIEVAL_QUERY

def debug_expansion():
    print("Initializing Models...")
    try:
        _, retriever_llm, embedder = initialize_models()
        print("Models initialized.")
        
        retriever = FallbackRetriever(
            driver=neo4j_driver, 
            index_name="documentsVectorIndex",
            retrieval_query=RETRIEVAL_QUERY,
            embedder=embedder,
            llm=retriever_llm
        )
        print("Retriever initialized.")
        
        print("Retriever initialized.")
        
        queries = [
            "What comes after the number 1?",
            "Which major international airport in south-east England ranks as the 8th busiest airport in Europe and replaced Croydon Airport?",
            "What is the capital of France?"
        ]

        for query in queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print(f"{'='*50}")
            
            try:
                print("\nExecuting Search...")
                results = retriever.search(query, top_k=5)
                
                print("\n--- Search Results ---")
                for item in results.items:
                    print(f"[{item.metadata.get('score', 0):.4f}] {item.content[:50]}...")
                    if "fax number" in item.content.lower():
                        print("!! FAILURE: 'fax number' found in results.")
                    
            except Exception as e:
                print(f"ERROR calling search: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"Initialization Error: {e}")

if __name__ == "__main__":
    debug_expansion()
