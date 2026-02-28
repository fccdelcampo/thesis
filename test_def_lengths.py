from neo4j import GraphDatabase
import statistics

# Connect to the local Neo4j instance
uri = "bolt://localhost:7687"
username = "neo4j"
password = "password"  # Replace with actual password if different, or we can check config.py

from prolog_graphrag_pipeline.graphrag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def analyze_definition_lengths(driver):
    with driver.session(database="neo4j") as session:
        result = session.run("MATCH (n:KBPediaConcept) WHERE n.definition IS NOT NULL RETURN n.definition AS def")
        lengths = [len(record["def"]) for record in result]
        
    if not lengths:
        print("No definitions found.")
        return
        
    avg_len = statistics.mean(lengths)
    median_len = statistics.median(lengths)
    max_len = max(lengths)
    min_len = min(lengths)
    total_defs = len(lengths)
    
    print(f"Total concepts with definitions: {total_defs}")
    print(f"Average definition length: {avg_len:.2f} characters")
    print(f"Median definition length:  {median_len} characters")
    print(f"Max definition length:     {max_len} characters")
    print(f"Min definition length:     {min_len} characters")
    
    # Let's see how many are over 200 chars (the old cutoff)
    over_200 = sum(1 for l in lengths if l > 200)
    print(f"\nConcepts exceeding old 200-char limit: {over_200} ({(over_200/total_defs)*100:.1f}%)")

if __name__ == "__main__":
    analyze_definition_lengths(driver)
    driver.close()
