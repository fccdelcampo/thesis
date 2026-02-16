from neo4j import GraphDatabase
import os

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "graphrag")

def check_yago_embeddings():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session() as session:
        result = session.run("MATCH (n:Resource) WHERE n.embedding IS NOT NULL RETURN count(n) as count")
        count = result.single()["count"]
        print(f"YAGO Nodes with Embeddings: {count}")
        
        result2 = session.run("MATCH (n:Resource) RETURN count(n) as total")
        total = result2.single()["total"]
        print(f"Total YAGO Nodes: {total}")
        
    driver.close()

if __name__ == "__main__":
    check_yago_embeddings()
