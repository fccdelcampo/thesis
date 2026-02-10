
from neo4j import GraphDatabase
from prolog_graphrag_pipeline.graphrag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def check_labels():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    with driver.session() as session:
        print("Checking Chunks and connected entities:")
        result = session.run("""
            MATCH (c:Chunk)
            RETURN count(c) as chunk_count
        """)
        print(f"Total Chunks: {result.single()['chunk_count']}")

        result = session.run("""
            MATCH (c:Chunk)-[:EXTRACTS]->(e)
            RETURN labels(e) as labels, e.name as name
            LIMIT 10
        """)
        found = False
        for record in result:
            found = True
            print(f"Entity: {record['name']}, Labels: {record['labels']}")
        
        if not found:
            print("No entities connected to Chunks via :EXTRACTS found.")
            
    driver.close()

if __name__ == "__main__":
    check_labels()
