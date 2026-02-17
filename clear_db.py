import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from prolog_graphrag_pipeline.graphrag.graphrag_driver import neo4j_driver

def clear_database():
    print("WARNING: This will delete ALL data in the Neo4j database.")
    confirm = input("Type 'yes' to confirm: ")
    if confirm != "yes":
        print("Aborted.")
        return

    print("Connecting to Neo4j...")
    try:
        with neo4j_driver.session() as session:
            print("Deleting all nodes and relationships...")
            session.run("MATCH (n) DETACH DELETE n")
            
            print("Dropping indexes...")
            # Drop indexes if they exist (best effort)
            try:
                session.run("DROP INDEX documentsVectorIndex IF EXISTS")
                session.run("DROP INDEX documentsFulltextIndex IF EXISTS")
                session.run("DROP INDEX yagoNameIndex IF EXISTS")
            except Exception as e:
                print(f"Index drop warning: {e}")
                
        print("Database cleared successfully.")
    except Exception as e:
        print(f"Error clearing database: {e}")

if __name__ == "__main__":
    clear_database()
