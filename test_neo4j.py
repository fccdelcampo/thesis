from neo4j import GraphDatabase
import sys

def test_connect(uri):
    print(f"Testing connection to {uri}...", flush=True)
    try:
        driver = GraphDatabase.driver(uri, auth=("neo4j", "graphrag"))
        driver.verify_connectivity()
        print("Success!", flush=True)
        driver.close()
    except Exception as e:
        print(f"Failed: {e}", flush=True)

if __name__ == "__main__":
    test_connect("bolt://127.0.0.1:7687")
    # test_connect("neo4j://127.0.0.1:7687") 
