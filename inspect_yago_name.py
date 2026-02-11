import neo4j

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def find_name_prop():
    print("--- Finding Name Property ---")
    # Identify a node with many properties and print all keys/values
    # We filter by having 'uri' to ensure it's a YAGO node
    query = """
    MATCH (n)
    WHERE n.uri IS NOT NULL
    RETURN keys(n) as keys, properties(n) as props
    LIMIT 1
    """
    results = driver.execute_query(query)
    for r in results.records:
        props = r['props']
        for k, v in props.items():
            print(f"Key: {k} | Value: {v} | Type: {type(v)}")

find_name_prop()
driver.close()
