import neo4j

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def check_label():
    print("--- Checking for rdfs__label ---")
    query = """
    MATCH (n)
    WHERE n.rdfs__label IS NOT NULL
    RETURN n.rdfs__label as label, n.uri as uri
    LIMIT 5
    """
    results = driver.execute_query(query)
    if not results.records:
        print("No nodes with 'rdfs__label' found. Checking for 'skos__prefLabel'...")
        q2 = "MATCH (n) WHERE n.skos__prefLabel IS NOT NULL RETURN n.skos__prefLabel LIMIT 5"
        r2 = driver.execute_query(q2)
        if not r2.records:
             print("No skos__prefLabel either.")
             # Check for just 'label'
             q3 = "MATCH (n) WHERE n.label IS NOT NULL RETURN n.label LIMIT 5"
             r3 = driver.execute_query(q3)
             for r in r3.records: print(f"Found 'label': {r['label']}")
        else:
             for r in r2.records: print(f"Found 'skos__prefLabel': {r['skos__prefLabel']}")
    else:
        for r in results.records:
            print(f"Found 'rdfs__label': {r['label']}")

check_label()
driver.close()
