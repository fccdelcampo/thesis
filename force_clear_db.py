
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from prolog_graphrag_pipeline.graphrag.graphrag_driver import neo4j_driver

def force_clear_database():
    print("Force clearing Neo4j database...")
    try:
        with neo4j_driver.session(database="neo4j") as session:
            # Count before
            result = session.run("MATCH (n) RETURN count(n) as count")
            count_before = result.single()["count"]
            print(f"Nodes before clearing: {count_before}")

            # Delete all nodes and relationships
            print("Deleting all nodes and relationships...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Drop indexes if they exist
            print("Dropping indexes...")
            indexes = ["documentsVectorIndex", "documentsFulltextIndex", "yagoNameIndex"]
            for idx in indexes:
                try:
                    session.run(f"DROP INDEX {idx} IF EXISTS")
                    print(f"Dropped {idx}")
                except Exception as e:
                    print(f"Failed to drop {idx}: {e}")
            
            # Count after
            result = session.run("MATCH (n) RETURN count(n) as count")
            count_after = result.single()["count"]
            print(f"Nodes after clearing: {count_after}")
            
            if count_after == 0:
                print("Database cleared successfully.")
            else:
                print("WARNING: Database not fully cleared!")

    except Exception as e:
        print(f"Error clearing database: {e}")

if __name__ == "__main__":
    force_clear_database()
