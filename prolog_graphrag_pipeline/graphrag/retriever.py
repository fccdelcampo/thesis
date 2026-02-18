from .config import RETRIEVAL_QUERY, RETRIEVER
from typing import List, Any
# CHANGE 1: Use the standard HybridRetriever (not HybridCypherRetriever)
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.retrievers import VectorCypherRetriever
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


# YAGO index creation removed — YAGO is no longer used as a knowledge source.

def create_retriever(driver, embedder):
    if not check_db_health(driver):
        raise ConnectionError("Aborting initialization: Cannot connect to Neo4j.")
    
    # 2. Cleanup Phase (Only for local document indexes)
    driver.execute_query("DROP INDEX documentsVectorIndex IF EXISTS")
    driver.execute_query("DROP INDEX documentsFulltextIndex IF EXISTS")
    
    # 3. (YAGO index removed — no longer needed)

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
    elif RETRIEVER.value == "VectorCypherRetriever":
        return EnhancedRetriever(
            driver=driver,
            index_name="documentsVectorIndex",
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
    def __init__(self, driver, index_name, retrieval_query, embedder, llm=None):
        super().__init__(driver=driver, index_name=index_name, retrieval_query=retrieval_query, embedder=embedder)
        self.driver = driver
        self.llm = llm

    def expand_query(self, query):
        if not self.llm:
            return query
        
        prompt = f"""
        ### Task
        Analyze the user's query and generate a list of **Specific, Distinct Concepts** or **Entities** for a Knowledge Graph lookup.
        
        ### User Query
        "{query}"
        
        ### Instructions
        1. Identify the core domain (e.g., "Number Theory").
        2. List 3-5 specific concepts (e.g., "Integer", "Successor function").
        3. **CRITICAL**: Exclude generic words like "Number", "Line", "System", "of", "the" unless they are part of a highly specific phrase (e.g. "Number Theory" is okay, "Number" is not).
        4. **CRITICAL**: EXCLUDE irrelevant terms (e.g. "fax", "phone").
        5. Format the output as a JSON object with a list of strings "concepts".
        
        ### Example
        Query: "What comes after 1?"
        {{
            "concepts": ["Integer", "Successor function", "Natural number", "Counting"]
        }}
        
        ### Output (JSON ONLY)
        """
        try:
            response_text = self.llm.invoke(prompt).content.strip()
            # Clean possible markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            import json
            data = json.loads(response_text)
            concepts = data.get("concepts", [])
            
            # Filter concepts
            filtered_concepts = []
            stop_words = {'of', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'what', 'next', 'number'}
            for c in concepts:
                # Remove stopwords from within the concept string if it's just a stopword (unlikely) or if we want to be strict
                # Actually, simply quoting them is safer IF they are good concepts.
                # But LLM might output "next integer". "next" is generic.
                # Let's trust the LLM's "Specific" instruction but use quotes.
                if c.lower() not in stop_words:
                    filtered_concepts.append(f'"{c}"')

            if filtered_concepts:
                lucene_query = " OR ".join(filtered_concepts)
                print(f"DEBUG: LLM Expanded Concepts: {concepts}")
                print(f"DEBUG: Lucene Query: '{lucene_query}'")
                return lucene_query
            
            # If filtration resulted in empty list, use cleaned original query
            print("DEBUG: Expansion resulted in empty concepts (all filtered). Using cleaned query.")
            stop_words = {'what', 'is', 'the', 'capital', 'of', 'who', 'where', 'when', 'how', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'number'}
            keywords = [word for word in query.replace('?','').split() if word.lower() not in stop_words]
            return " ".join(keywords) if keywords else query

        except Exception as e:
            print(f"DEBUG: Query expansion failed: {e}. Falling back to cleaned query.")
            stop_words = {'what', 'is', 'the', 'capital', 'of', 'who', 'where', 'when', 'how', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'number'}
            keywords = [word for word in query.replace('?','').split() if word.lower() not in stop_words]
            return " ".join(keywords) if keywords else query

    def generate_wikidata_queries(self, query: str) -> List[str]:
        """
        Uses LLM to extract specific entities/concepts with minimal context.
        Avoids broad topic expansion (e.g. 'Number Theory') in favor of precise entity targeting (e.g. '1 (number)').
        """
        if not self.llm:
            return []
            
        prompt = f"""
        ### Task
        Extract the **Main Entities** from the user's query for a Wikidata search.
        
        ### Rules
        1. **Specific Entities**: Extract the names of people, places, objects, or specific concepts mentioned in the **Question Body**.
        2. **IGNORE Options**: If the query contains multiple-choice options (e.g., A), B), I., II.), IGNORE THEM completely. Do NOT extract terms from the answer choices.
        3. **Disambiguate if needed**: If an entity is a common word or number (e.g., "1", "Apple"), include its **immediate context** from the query to verify what it is.
        4. **No Abstract Topics**: Do NOT infer broad academic fields like "Number Theory" or "Mathematics" unless explicitly mentioned.
        5. **No Synonyms**: Use the exact words from the query where possible.
        
        ### User Query
        "{query}"
        
        ### Output Format (JSON)
        {{
            "queries": ["ENTITY_1", "ENTITY_2 (CONTEXT)"]
        }}
        """
        try:
            response_text = self.llm.invoke(prompt).content.strip()
            # Clean possible markdown field
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            import json
            data = json.loads(response_text)
            return data.get("queries", [])
        except Exception as e:
            print(f"DEBUG: Smart selection failed: {e}")
            return []

    def search(self, query_text, top_k=5, **kwargs):
        # 1. Smart Entity Extraction & Wikidata Search (Global Knowledge)
        wikidata_items = []
        if self.llm:
            smart_terms = self.generate_wikidata_queries(query_text)
            if smart_terms:
                print(f"DEBUG: Triggering Wikidata Search for: {smart_terms}")
                if not hasattr(self, 'wikidata_retriever'):
                     from .wikidata_retriever import WikidataRetriever
                     self.wikidata_retriever = WikidataRetriever()
                
                # Use " OR " to search for any of the terms
                wd_search_query = " OR ".join([f'"{t}"' for t in smart_terms])
                wd_result = self.wikidata_retriever.search(wd_search_query)
                if wd_result and wd_result.items:
                    wikidata_items = wd_result.items
                    print(f"DEBUG: Wikidata returned {len(wikidata_items)} items.")

        # 2. Local Vector Search (Local Documents)
        local_items = []
        MIN_SCORE = 0.4  # Minimum relevance threshold to filter noise
        MIN_CONTENT_LENGTH = 20  # Reject chunks with very short content
        try:
            # We want to use the original query context for vector search
            local_result = super().search(query_text=query_text, top_k=top_k, **kwargs)
            
            # Post-process results to clean content and fix scores
            if local_result and local_result.items:
                for item in local_result.items:
                    # Fix Content: Unwrap if it is a Neo4j Record string representation
                    if item.metadata and 'local_context' in item.metadata:
                        local_ctx = item.metadata['local_context']
                        if local_ctx and isinstance(local_ctx, list) and len(local_ctx) > 0:
                            clean_text = local_ctx[0].get('text')
                            if clean_text:
                                item.content = clean_text
                                
                    # If metadata doesn't have it, try regex fallback
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
                    
                    # Skip chunks with scores below threshold
                    if score < MIN_SCORE:
                        print(f"DEBUG: Filtered out low-score chunk ({score:.4f}): '{content[:50]}...'")
                        continue
                    
                    # Skip chunks with very short content (likely noise like 'In mathematics')
                    if len(content) < MIN_CONTENT_LENGTH:
                        print(f"DEBUG: Filtered out short chunk ({len(content)} chars): '{content}'")
                        continue
                    
                    filtered_items.append(item)
                
                local_items = filtered_items
                print(f"DEBUG: Local Vector Search returned {len(local_items)} chunks (after filtering).")

        except Exception as e:
            print(f"DEBUG: Main retriever failed: {e}")
            
        # 3. Merge and Rerank
        # We prioritize Wikidata Exact Matches (Score ~1.0) but also want Semantic Context from Local.
        # We will interleave or just sort by score?
        # Given "In mathematics" might have high semantic similarity but low relevance to "Heathrow", result merging is key.
        
        # Combine lists
        all_items = wikidata_items + local_items
        
        # Sort descending by score
        # Ensure every item has a score
        for item in all_items:
            if 'score' not in item.metadata:
                item.metadata['score'] = 0.0
                
        # Sort by score descending
        all_items.sort(key=lambda x: x.metadata.get('score', 0.0), reverse=True)
        
        # Deduplication (by content or ID?) - simplistic for now
        seen_content = set()
        unique_items = []
        for item in all_items:
            # Use a snippet of content as signature
            sig = item.content[:50]
            if sig not in seen_content:
                seen_content.add(sig)
                unique_items.append(item)
        
        print(f"DEBUG: Merged Result Count: {len(unique_items)}")
        
        # Return combined results. We increase top_k slightly to handle the mix.
        # If we have BOTH local and global, we want to show both.
        final_top_k = top_k * 2 if (wikidata_items and local_items) else top_k
        
        return RetrieverResult(items=unique_items[:final_top_k], metadata={})

def generate(llm, retriever, query) -> dict:
    # If the retriever is our EnhancedRetriever, inject the LLM
    if isinstance(retriever, EnhancedRetriever):
        retriever.llm = llm

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
            # Fix Score: config.py now puts 'score' in metadata
            metadata = item.metadata or {}
            score = metadata.get('score', 0.0) 

            # Add Metadata Dump (requested by user)
            import json
            f.write(f"\n[Metadata Dump]\n{json.dumps(metadata, indent=2)}\n") 
            
            # Fix Content: Unwrap if it is a Neo4j Record string representation or just use metadata text
            # If item.content looks like "<Record ...>", we prefer metadata['text'] or parse it.
            # However, our query returns `node.text as text`.
            # Let's inspect content. If it starts with "<Record", we try to clean it.
            clean_content = item.content
            if clean_content and "<Record" in str(clean_content):
                # Try to extract text='...'
                import re
                match = re.search(r"text='(.*?)'", clean_content)
                if match:
                    clean_content = match.group(1)
                else:
                    # Fallback: Check if local_context is present and has text
                    local_ctx = metadata.get('local_context', [])
                    if local_ctx and len(local_ctx) > 0:
                         clean_content = local_ctx[0].get('text', clean_content)
                    
            clean_content = clean_content.replace('\n', ' ').replace('  ', ' ')
            
            # Extract Local/Global Context
            local_ctx = metadata.get('local_context', [])
            global_ctx = metadata.get('global_context', [])
            
            enriched_content = f"[Chunk Score: {score:.4f}] {clean_content}\n"
            
            if local_ctx:
                enriched_content += "  [Local Connections]:\n"
                for lc in local_ctx:
                    enriched_content += f"   - {lc.get('entity')} {lc.get('relationship')} {lc.get('target')}\n"
            
            # Check for knowledge graph triples (from Wikidata/VectorCypherRetriever)
            kg_triples = metadata.get('triples', [])
            if kg_triples:
                 enriched_content += "  [Knowledge Graph Triples]:\n"
                 for t in kg_triples:
                     enriched_content += f"   - {t}\n"

            if global_ctx:
                enriched_content += "  [Global Knowledge]:\n"
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