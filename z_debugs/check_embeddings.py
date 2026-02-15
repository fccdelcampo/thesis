
from neo4j import GraphDatabase
from prolog_graphrag_pipeline.graphrag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def check_embeddings():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    with driver.session() as session:
        print("Checking Chunks for embeddings:")
        result = session.run("""
            MATCH (c:Chunk)
            RETURN c.id as id, c.text as text, size(c.embedding) as dim
            LIMIT 5
        """)
        found = False
        for record in result:
            found = True
            print(f"Chunk ID: {record['id']}, Dim: {record['dim']}, Text Preview: {record['text'][:50]}...")
        
        if not found:
            print("No Chunks found.")

    driver.close()

if __name__ == "__main__":
    check_embeddings()
