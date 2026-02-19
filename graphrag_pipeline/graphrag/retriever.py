from .config import RETRIEVAL_QUERY, RETRIEVER
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.retrievers import VectorCypherRetriever
import time
from neo4j.exceptions import ServiceUnavailable, AuthError
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from typing import List

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


from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem

# Helper classes for Fallback
class SimpleItem:
    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}

class SimpleResult:
    def __init__(self, items):
        self.items = items
        self.metadata = {}


class EnhancedRetriever(VectorCypherRetriever):
    """
    Extended retriever that merges local Neo4j vector search results
    with Wikidata global knowledge for richer context.
    Also filters out low-quality/noise chunks.
    """
    def __init__(self, driver, index_name, retrieval_query, embedder, llm=None):
        super().__init__(driver=driver, index_name=index_name, retrieval_query=retrieval_query, embedder=embedder)
        self.driver = driver
        self.llm = llm

    def generate_wikidata_queries(self, query: str) -> dict:
        """
        Uses LLM to extract specific entities/concepts for Wikidata search.
        Returns: {"queries": [...]}
        """
        if not self.llm:
            return {"queries": []}
            
        prompt = f"""
        ### Task
        Extract the **Main Entities and Concepts** from the user's educational query for a Wikidata search.
        
        ### Rules
        1. **Specific Entities**: Extract names of people, places, theories, formulas, or specific concepts.
        2. **Disambiguate**: If an entity is ambiguous (e.g., "Apple"), include context (e.g., "Apple company" or "Apple fruit").
        3. **Concepts**: Include scientific terms, mathematical concepts, or literary works if they are the focus.
        4. **No Synonyms**: Use the exact words from the query where possible.
        
        ### User Query
        "{query}"
        
        ### Output Format (JSON)
        {{
            "queries": ["ENTITY_1", "ENTITY_2 (CONTEXT)"]
        }}
        """
        try:
            response_text = self.llm.invoke(prompt).content.strip()
            # Clean possible markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            import json
            data = json.loads(response_text)
            return {
                "queries": data.get("queries", [])
            }
        except Exception as e:
            print(f"DEBUG: Smart selection failed: {e}")
            return {"queries": []}

    def search(self, query_text, top_k=5, **kwargs):
        # 1. Smart Entity Extraction & Wikidata Search (Global Knowledge)
        wikidata_items = []
        if self.llm:
            extraction_result = self.generate_wikidata_queries(query_text)
            smart_terms = extraction_result.get("queries", [])
            
            if smart_terms:
                print(f"DEBUG: Triggering Wikidata Search for: {smart_terms}")
                if not hasattr(self, 'wikidata_retriever'):
                    # Import from the prolog pipeline which has the shared wikidata_retriever
                    from prolog_graphrag_pipeline.graphrag.wikidata_retriever import WikidataRetriever
                    self.wikidata_retriever = WikidataRetriever()
                
                # Use " OR " to search for any of the terms
                wd_search_query = " OR ".join([f'"{t}"' for t in smart_terms])
                # Search using Unified Educational Property Set (no category needed)
                wd_result = self.wikidata_retriever.search(wd_search_query)
                if wd_result and wd_result.items:
                    wikidata_items = wd_result.items
                    print(f"DEBUG: Wikidata returned {len(wikidata_items)} items.")

        # 2. Local Vector Search (Local Documents)
        local_items = []
        MIN_SCORE = 0.4  # Minimum relevance threshold to filter noise
        MIN_CONTENT_LENGTH = 20  # Reject chunks with very short content
        try:
            local_result = super().search(query_text=query_text, top_k=top_k, **kwargs)
            
            if local_result and local_result.items:
                for item in local_result.items:
                    # Fix Content: Unwrap if it is a Neo4j Record string representation
                    if item.metadata and 'local_context' in item.metadata:
                        local_ctx = item.metadata['local_context']
                        if local_ctx and isinstance(local_ctx, list) and len(local_ctx) > 0:
                            clean_text = local_ctx[0].get('text')
                            if clean_text:
                                item.content = clean_text
                                
                    elif item.content and "<Record" in str(item.content):
                        try:
                           import re
                           match = re.search(r"text='(.*?)'", item.content)
                           if match:
                               item.content = match.group(1)
                        except:
                            pass
                
                # Filter out low-quality chunks
                filtered_items = []
                for item in local_result.items:
                    score = item.metadata.get('score', 0.0) if item.metadata else 0.0
                    content = (item.content or "").strip()
                    
                    if score < MIN_SCORE:
                        print(f"DEBUG: Filtered out low-score chunk ({score:.4f}): '{content[:50]}...'")
                        continue
                    
                    if len(content) < MIN_CONTENT_LENGTH:
                        print(f"DEBUG: Filtered out short chunk ({len(content)} chars): '{content}'")
                        continue
                    
                    filtered_items.append(item)
                
                local_items = filtered_items
                print(f"DEBUG: Local Vector Search returned {len(local_items)} chunks (after filtering).")

        except Exception as e:
            print(f"DEBUG: Main retriever failed: {e}")
            
        # 3. Merge and Rerank
        all_items = wikidata_items + local_items
        
        for item in all_items:
            if 'score' not in item.metadata:
                item.metadata['score'] = 0.0
                
        all_items.sort(key=lambda x: x.metadata.get('score', 0.0), reverse=True)
        
        # Deduplication
        seen_content = set()
        unique_items = []
        for item in all_items:
            sig = item.content[:50]
            if sig not in seen_content:
                seen_content.add(sig)
                unique_items.append(item)
        
        print(f"DEBUG: Merged Result Count: {len(unique_items)}")
        
        final_top_k = top_k * 2 if (wikidata_items and local_items) else top_k
        
        return RetrieverResult(items=unique_items[:final_top_k], metadata={})


def create_retriever(driver, embedder):
    if not check_db_health(driver):
        raise ConnectionError("Aborting initialization: Cannot connect to Neo4j.")
    
    # 2. Cleanup Phase
    driver.execute_query("DROP INDEX documentsVectorIndex IF EXISTS")
    driver.execute_query("DROP INDEX documentsFulltextIndex IF EXISTS")
    
    # 3. Recreate Phase
    create_indexes(driver)
    
    # 4. Guarded Wait Phase
    try:
        wait_for_indexes(driver, timeout=120) 
    except TimeoutError as e:
        print(f"Critical Error: {e}")
        return None

    # 5. Initialization
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
    elif RETRIEVER.value == "VectorCypherRetriever":
        return EnhancedRetriever(
            driver=driver,
            index_name="documentsVectorIndex",
            retrieval_query=RETRIEVAL_QUERY,
            embedder=embedder
        )
    else:
        raise ValueError(f"Retriever {RETRIEVER.name} is invalid.")

def get_retriever(driver, embedder):
    """
    Instantiates the retriever without creating/dropping indexes.
    Assumes indexes already exist.
    """
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
    elif RETRIEVER.value == "VectorCypherRetriever":

        return EnhancedRetriever(
            driver=driver,
            index_name="documentsVectorIndex",

            retrieval_query=RETRIEVAL_QUERY,
            embedder=embedder
        )
    else:
        raise ValueError(f"Retriever {RETRIEVER.name} is invalid.")