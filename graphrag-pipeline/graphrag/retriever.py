from .config import RETRIEVAL_QUERY, GRAPHRAG_TEMPLATE
from neo4j_graphrag.retrievers import HybridCypherRetriever
import time
from neo4j.exceptions import ServiceUnavailable, AuthError
from neo4j_graphrag.generation import GraphRAG

def check_db_health(driver):
    """
    Verifies the driver can connect to Neo4j before starting the pipeline.
    """
    try:
        # Test the credentials and database availability
        driver.verify_connectivity()
        print("Pre-flight check: Database connection verified.")
        return True
    except (ServiceUnavailable, AuthError) as e:
        print(f"Pre-flight check failed: {e}")
        return False

def create_indexes(driver):
    driver.execute_query("""
        CREATE VECTOR INDEX documentsVectorIndex IF NOT EXISTS
        FOR (n:Chunk) ON (n.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 4096,
          `vector.similarity_function`: 'cosine'
        }}
    """, database_="neo4j")
    
    driver.execute_query("""
        CREATE FULLTEXT INDEX documentsFulltextIndex IF NOT EXISTS 
        FOR (n:Chunk) ON EACH [n.text]
    """, database_="neo4j")
    print("Index creation commands sent.")

def wait_for_indexes(driver, timeout=60):
    """
    Waits for indexes to go ONLINE, with a maximum wait time.
    """
    target_indexes = ['documentsVectorIndex', 'documentsFulltextIndex']
    start_time = time.time()
    
    while True:
        # Check if we have exceeded the timeout duration
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Indexes failed to go ONLINE within {timeout} seconds.")
            
        records, _, _ = driver.execute_query(
            "SHOW INDEXES YIELD name, state WHERE name IN $names",
            names=target_indexes,
            database_="neo4j"
        )
        
        states = [record["state"] for record in records]
        if len(states) == 2 and all(s == "ONLINE" for s in states):
            print("Database is ready for retrieval.")
            break
            
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed}s] Waiting for indexes... Status: {states}")
        time.sleep(2)

def create_retriever(driver, embedder):
    # 1. Immediate Health Check
    if not check_db_health(driver):
        raise ConnectionError("Aborting initialization: Cannot connect to Neo4j.")

    # 2. Reset Phase
    print("Clearing old indexes...")
    driver.execute_query("DROP INDEX documentsVectorIndex IF EXISTS")
    driver.execute_query("DROP INDEX documentsFulltextIndex IF EXISTS")
    
    # 3. Recreate Phase
    create_indexes(driver)
    
    # 4. Guarded Wait Phase (with 2-minute timeout for large graphs)
    try:
        wait_for_indexes(driver, timeout=120) 
    except TimeoutError as e:
        print(f"Critical Error: {e}")
        return None

    # 5. Initialization
    return HybridCypherRetriever(
        driver=driver,
        vector_index_name="documentsVectorIndex",
        fulltext_index_name="documentsFulltextIndex",
        retrieval_query=RETRIEVAL_QUERY,
        embedder=embedder
    )