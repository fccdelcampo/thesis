import neo4j

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def inspect_schema():
    print("--- Inspecting Node Labels ---")
    # Get top 10 labels
    labels = driver.execute_query("CALL db.labels() YIELD label RETURN label LIMIT 10")
    print([r['label'] for r in labels.records])

    print("\n--- Inspecting Relationships ---")
    rels = driver.execute_query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType LIMIT 10")
    print([r['relationshipType'] for r in rels.records])

    print("\n--- Sample YAGO Node ---")
    # Try to find a node that likely came from YAGO (assuming it has a URI or specific label)
    # We'll look for something that is NOT a Chunk or Entity
    sample = driver.execute_query("""
        MATCH (n) 
        WHERE NOT n:Chunk AND NOT n:Entity 
        RETURN labels(n) as labels, keys(n) as properties, n LIMIT 1
    """)
    for r in sample.records:
        print(f"Labels: {r['labels']}")
        print(f"Properties: {r['properties']}")
        print(f"Node: {r['n']}")

    print("\n--- Checking N10s Configuration ---")
    try:
        config = driver.execute_query("MATCH (n:GraphConfig) RETURN n")
        for r in config.records:
            print(r['n'])
    except Exception as e:
        print(f"Could not read config: {e}")

inspect_schema()
driver.close()
