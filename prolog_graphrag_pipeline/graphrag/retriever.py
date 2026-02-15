from .config import RETRIEVAL_QUERY, RETRIEVER, YAGO_FALLBACK_QUERY
# CHANGE 1: Use the standard HybridRetriever (not HybridCypherRetriever)
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.retrievers import HybridCypherRetriever
import time
from neo4j.exceptions import ServiceUnavailable, AuthError
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from .config import GRAPHRAG_TEMPLATE
from neo4j_graphrag.generation import GraphRAG
from langchain_core.documents import Document # Ensure this is available

# ... (db health check omitted)

# ... (create_yago_index omitted)

# ... (create_retriever omitted)

def check_db_health(driver):
    """Verifies the driver can connect to Neo4j."""
    try:
        driver.verify_connectivity()
        print("Pre-flight check: Database connection verified.")
        return True
    except (ServiceUnavailable, AuthError) as e:
        print(f"Pre-flight check failed: {e}")
        return False

def create_indexes(driver):
    embedding_dimension = 4096 

    print(f"Creating indexes with dimension: {embedding_dimension}...")

    driver.execute_query(f"""
        CREATE VECTOR INDEX documentsVectorIndex IF NOT EXISTS
        FOR (n:Chunk) ON (n.embedding)
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: {embedding_dimension},
          `vector.similarity_function`: 'cosine'
        }}}}
    """, database_="neo4j")
    
    driver.execute_query("""
        CREATE FULLTEXT INDEX documentsFulltextIndex IF NOT EXISTS 
        FOR (n:Chunk) ON EACH [n.text]
    """, database_="neo4j")
    print("Index creation commands sent.")

def wait_for_indexes(driver, timeout=60):
    target_indexes = ['documentsVectorIndex', 'documentsFulltextIndex']
    start_time = time.time()
    
    while True:
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

class CleanTextRetriever(HybridRetriever):
    # We override 'search' (Public API) instead of 'get_search_results' (Internal API)
    def search(self, query_text, top_k=5, **kwargs):
        result = super().search(query_text=query_text, top_k=top_k, **kwargs)
        
        # Now it is safe to iterate over .items
        for item in result.items:
            if item.content:
                # Clean the text
                item.content = item.content.replace('\n', ' ').replace('  ', ' ')
        
        return result


def create_yago_index(driver):
    print("Ensuring YAGO Fulltext Index exists...")
    try:
        driver.execute_query("""
            CREATE FULLTEXT INDEX yagoNameIndex IF NOT EXISTS 
            FOR (n:Resource) ON EACH [n.rdfs__label]
        """, database_="neo4j")
        print("YAGO index check complete.")
    except Exception as e:
        print(f"Warning: Could not create YAGO index: {e}")

def create_retriever(driver, embedder):
    if not check_db_health(driver):
        raise ConnectionError("Aborting initialization: Cannot connect to Neo4j.")
    
    # 2. Cleanup Phase (Only for local document indexes)
    driver.execute_query("DROP INDEX documentsVectorIndex IF EXISTS")
    driver.execute_query("DROP INDEX documentsFulltextIndex IF EXISTS")
    
    # 3. YAGO Index creation (Static, do not drop)
    create_yago_index(driver)

    # 4. Recreate Phase
    create_indexes(driver)
    
    # 5. Guarded Wait Phase
    try:
        wait_for_indexes(driver, timeout=120) 
    except TimeoutError as e:
        print(f"Critical Error: {e}")
        return None

    # 6. Initialization
    if RETRIEVER.value == "HybridRetriever":
        return CleanTextRetriever(
            driver=driver,
            vector_index_name="documentsVectorIndex",
            fulltext_index_name="documentsFulltextIndex",
            embedder=embedder,
            return_properties=["text", "id"] 
        )
    elif RETRIEVER.value == "HybridCypherRetriever":
        return HybridCypherRetriever(
            driver=driver,
            vector_index_name="documentsVectorIndex",
            fulltext_index_name="documentsFulltextIndex",
            retrieval_query=RETRIEVAL_QUERY,
            embedder=embedder
        )
    else:
        raise ValueError(f"Retriever {RETRIEVER.name} is invalid.")
    
    # Monkey-patch driver if not present
    if not hasattr(r, 'driver'):
        r.driver = driver
    return r


from neo4j_graphrag.retrievers.base import Retriever

# Helper classes for Fallback
class SimpleItem:
    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}

class SimpleResult:
    def __init__(self, items):
        self.items = items
        self.metadata = {}

class FallbackRetriever(Retriever):
    def __init__(self, main_retriever):
        super().__init__(driver=getattr(main_retriever, 'driver', None)) # Base init might expect driver
        self.main = main_retriever
        self.driver = getattr(main_retriever, 'driver', None)

    def search(self, query_text, top_k=5, **kwargs):
        # 1. Try Main Retriever
        try:
            result = self.main.search(query_text=query_text, top_k=top_k, **kwargs)
        except Exception as e:
            print(f"DEBUG: Main retriever failed: {e}")
            result = None
            
        # 2. Check if successful
        if result and result.items and len(result.items) > 0:
            return result
            
        # 3. Fallback to YAGO
        print("DEBUG: 0 Chunks encountered. Executing YAGO Fallback...")
        if not self.driver:
            print("ERROR: No driver found on retriever for fallback.")
            return SimpleResult([])

        try:
            with self.driver.session() as session:
                records = session.run(YAGO_FALLBACK_QUERY, query_text=query_text)
                items = []
                for record in records:
                    items.append(SimpleItem(
                        content=record['text'],
                        metadata=record['metadata']
                    ))
                print(f"DEBUG: YAGO Fallback found {len(items)} items.")
                return SimpleResult(items)
        except Exception as e:
            print(f"DEBUG: YAGO Fallback error: {e}")
            return SimpleResult([])

def generate(llm, retriever, query) -> dict:
    # Wrap in Fallback
    retriever = FallbackRetriever(retriever)

    rag_template = RagTemplate(template=GRAPHRAG_TEMPLATE, expected_inputs=['query_text', 'context'])
    
    graph_rag = GraphRAG(llm=llm, retriever=retriever, prompt_template=rag_template)

    graph_rag_results = graph_rag.search(
        query, 
        retriever_config={'top_k': 5}, 
        return_context=True
    )
    
    retrieved_context = graph_rag_results.retriever_result
    answer = graph_rag_results.answer

    # Enrich context with metadata for the LLM
    final_contexts = []
    
    output_file = "debug_rag_output.txt"
    
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(f" QUERY: {query}\n")
        f.write("="*50 + "\n\n")
        f.write(f"--- RETRIEVED CONTEXT (Top {len(retrieved_context.items)} Items) ---\n")
        
        for i, item in enumerate(retrieved_context.items):
            score = item.metadata.get('score', 0.0) if item.metadata else 0.0
            clean_content = item.content.replace('\n', ' ').replace('  ', ' ')
            
            # Extract Local/Global Context from new query structure if present
            metadata = item.metadata or {}
            local_ctx = metadata.get('local_context', [])
            global_ctx = metadata.get('global_context', [])
            
            enriched_content = f"[Chunk Score: {score:.4f}] {clean_content}\n"
            
            if local_ctx:
                enriched_content += "  [Local Connections]:\n"
                for lc in local_ctx:
                    enriched_content += f"   - {lc.get('entity')} {lc.get('relationship')} {lc.get('target')}\n"
            
            if global_ctx:
                enriched_content += "  [Global Knowledge (YAGO)]:\n"
                for gc in global_ctx:
                     enriched_content += f"   - {gc.get('concept')}: {gc.get('definition')} (Score: {gc.get('score'):.2f})\n"

            final_contexts.append(enriched_content)

            f.write(f"[Item {i+1}]\n")
            f.write(f"{enriched_content}\n")
            f.write("-" * 50 + "\n")

        f.write("\n" + "="*50 + "\n")
        f.write(" LLM ANSWER:\n")
        f.write("="*50 + "\n")
        f.write(f"{answer}\n")
        f.write("="*50 + "\n")

    print(f"Done. Check '{output_file}' for the context and answer.")

    return {
        "answer": answer,
        "contexts": final_contexts
    }