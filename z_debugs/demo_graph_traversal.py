import sys
import os
import neo4j
import time
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.retrievers import HybridCypherRetriever

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

try:
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Connected to Neo4j.")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")
    sys.exit(1)

embedder = OllamaEmbeddings(model="llama3.1:8b-instruct-q4_K_M")

# (Assumes setup_sample_data was run successfully by previous script version, 
#  so we skip re-creating data to save time, unless it failed last time. 
#  But let's just make sure indices are ready)
index_name = "sampleVectorIndex"
ft_index_name = "sampleFulltextIndex"


MY_RETRIEVAL_QUERY = """
MATCH (node)-[:EXTRACTS]->(e:Entity)
OPTIONAL MATCH (e)-[r]->(target:Entity)
RETURN node.text AS text, 
       node.score AS score,
       {
         visited_node: e.name, 
         traversed_relationship: type(r), 
         connected_to: target.name
       } AS metadata
"""

def run_retrieval():
    print("\n--- Initializing HybridCypherRetriever (with 'AS metadata') ---")
    retriever = HybridCypherRetriever(
        driver=driver,
        vector_index_name=index_name, 
        fulltext_index_name=ft_index_name,
        retrieval_query=MY_RETRIEVAL_QUERY,
        embedder=embedder
    )
    
    query_text = "Who is Socrates?"
    print(f"Querying: '{query_text}'")
    
    try:
        results = retriever.search(query_text=query_text, top_k=1)
        
        print("\n--- RESULTS ---")
        if not results.items:
            print("No results found.")
        
        for item in results.items:
            print(f"\n[Content]: {item.content}")
            print(f"[Item Attributes]: {dir(item)}") # Check availability
            
            # Check metadata
            if item.metadata:
                print(f"[Metadata]: {item.metadata}")
            else:
                print("[Metadata]: None")

    except Exception as e:
        print(f"Retrieval failed: {e}")

if __name__ == "__main__":
    try:
        run_retrieval()
    finally:
        driver.close()
