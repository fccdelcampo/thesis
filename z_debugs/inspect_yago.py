
from neo4j import GraphDatabase
from prolog_graphrag_pipeline.graphrag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def inspect_yago():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # Check labels of nodes with rdfs__label
        print("Checking labels for nodes with 'rdfs__label':")
        result = session.run("""
            MATCH (n) 
            WHERE n.rdfs__label IS NOT NULL 
            RETURN labels(n) as labels, count(*) as count 
            LIMIT 10
        """)
        for record in result:
            print(f"Labels: {record['labels']}, Count: {record['count']}")
            
        # Check if any indexes exist
        print("\nChecking existing indexes:")
        result = session.run("SHOW INDEXES")
        for record in result:
            print(f"Index: {record['name']}, Type: {record['type']}, Entity: {record['entityType']}, Check: {record['properties']}")

    driver.close()

if __name__ == "__main__":
    inspect_yago()
