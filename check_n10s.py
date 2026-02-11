import neo4j
import sys

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag"

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def check_n10s():
    try:
        # Try to call a n10s procedure
        result = driver.execute_query("CALL n10s.graph.config.init() YIELD param RETURN param LIMIT 1")
        print("n10s is INSTALLED and INITIALIZED.")
    except Exception as e:
        if "Procedure not found" in str(e):
             print("n10s is NOT INSTALLED or NOT LOADED.")
             print("Please install Neosemantics (n10s) plugin for your Neo4j version.")
        elif "ConstraintValidationFailed" in str(e) or "already exists" in str(e):
             print("n10s is INSTALLED (Config already exists).")
        else:
             print(f"Error checking n10s: {e}")

check_n10s()
driver.close()
