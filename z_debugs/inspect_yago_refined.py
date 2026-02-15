import neo4j

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def inspect_yago():
    print("--- Checking for nodes with 'uri' property (n10s default) ---")
    query = """
    MATCH (n)
    WHERE n.uri IS NOT NULL
    RETURN labels(n) as labels, keys(n) as properties, n.uri as uri, n.label as label_prop, n.name as name_prop
    LIMIT 5
    """
    results = driver.execute_query(query)
    if not results.records:
        print("No nodes with 'uri' property found.")
        
        print("\n--- Checking for ANY node that is not Chunk/Entity ---")
        query2 = """
        MATCH (n)
        WHERE NOT n:Chunk AND NOT n:Entity AND NOT n:__KGBuilder__ AND NOT n:Document
        RETURN labels(n) as labels, keys(n) as properties
        LIMIT 5
        """
        results2 = driver.execute_query(query2)
        for r in results2.records:
            print(r)
    else:
        for r in results.records:
            print(f"Labels: {r['labels']}")
            print(f"URI: {r['uri']}")
            print(f"Properties: {r['properties']}")

inspect_yago()
driver.close()
