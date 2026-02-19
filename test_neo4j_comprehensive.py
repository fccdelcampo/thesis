from neo4j import GraphDatabase
import sys

URIS = [
    "bolt://127.0.0.1:7687",
    "bolt+s://127.0.0.1:7687",
    "bolt+ssc://127.0.0.1:7687",
    "neo4j://127.0.0.1:7687",
    "neo4j+s://127.0.0.1:7687",
    "neo4j+ssc://127.0.0.1:7687"
]

def test_connect(uri):
    print(f"Testing connection to {uri}...", flush=True)
    try:
        driver = GraphDatabase.driver(uri, auth=("neo4j", "graphrag"), connection_timeout=5, max_connection_lifetime=5)
        driver.verify_connectivity()
        print(f"SUCCESS: {uri}", flush=True)
        driver.close()
        return True
    except Exception as e:
        print(f"FAILED: {uri} - {e}", flush=True)
        return False

if __name__ == "__main__":
    for uri in URIS:
        if test_connect(uri):
            print(f"Found working URI: {uri}")
            break
