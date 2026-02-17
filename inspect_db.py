
import sys
import os
from neo4j import GraphDatabase

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prolog_graphrag_pipeline.graphrag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def inspect_db():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        # Check one Chunk
        records, _, _ = driver.execute_query("""
            MATCH (n:Chunk) 
            RETURN n.embedding as emb
            LIMIT 1
        """)
        if records:
            emb = records[0]["emb"]
            start_type = type(emb)
            print(f"Found Chunk. Embedding Type: {start_type}")
            print(f"Embedding Sample: {str(emb)[:50]}...")
            
            if isinstance(emb, str):
                print("DETECTED: Imbedding is a STRING. Attempting to fix...")
                # Fix: Convert String to List<Float>
                # Using apoc.convert.fromJsonList if available, or just parse it.
                # Since we don't know if APOC is there, let's try a pure cypher parse if possible, 
                # but standard Cypher doesn't parse JSON arrays easily without APOC.
                # We will just report it for now.
    except Exception as e:
        print(f"Inspection failed: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    inspect_db()
