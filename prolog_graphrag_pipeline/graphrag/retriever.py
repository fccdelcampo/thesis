from typing import List, Any
import time
import asyncio
import json
import re
import types

from neo4j.exceptions import ServiceUnavailable, AuthError
from langchain_core.documents import Document

from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.retrievers import HybridRetriever, HybridCypherRetriever, VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem

from .config import RETRIEVAL_QUERY, RETRIEVER, GRAPHRAG_TEMPLATE

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

def _sanitize_lucene_query(query_text: str) -> str:
    """Escape Lucene special characters to prevent query parse errors."""
    # Lucene reserved chars: + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
    special_chars = r'+-&|!(){}[]^"~*?:\/'
    sanitized = []
    for ch in query_text:
        if ch in special_chars:
            sanitized.append(f'\\{ch}')
        else:
            sanitized.append(ch)
    return ''.join(sanitized)

def patched_hybrid_search(self, query_text, top_k=5, **kwargs):
    safe_query = _sanitize_lucene_query(query_text)
    result = self._original_search(query_text=safe_query, top_k=top_k, **kwargs)
    
    for item in result.items:
        if item.content:
            item.content = item.content.replace('\n', ' ').replace('  ', ' ')
    
    return result

def expand_query(self, query):
    if not hasattr(self, 'llm') or not self.llm:
        return query
    
    prompt = f"""
    ### Task
    Analyze the user's query and generate a list of **Specific, Distinct Concepts** or **Entities** for a Knowledge Graph lookup.
    
    ### User Query
    "{query}"
    
    ### Instructions
    1. Identify the core domain (e.g., "Number Theory").
    2. List 3-5 specific concepts (e.g., "Integer", "Successor function").
    3. **CRITICAL**: Exclude generic words like "Number", "Line", "System", "of", "the" unless they are part of a highly specific phrase.
    4. **CRITICAL**: EXCLUDE irrelevant terms (e.g. "fax", "phone").
    5. Format the output as a JSON object with a list of strings "concepts".
    """
    try:
        response_text = self.llm.invoke(prompt).content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        data = json.loads(response_text)
        concepts = data.get("concepts", [])
        
        filtered_concepts = []
        stop_words = {'of', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'what', 'next', 'number'}
        for c in concepts:
            if c.lower() not in stop_words:
                filtered_concepts.append(f'"{c}"')

        if filtered_concepts:
            lucene_query = " OR ".join(filtered_concepts)
            print(f"DEBUG: LLM Expanded Concepts: {concepts}")
            return lucene_query
        
        return query
    except Exception as e:
        print(f"DEBUG: Query expansion failed: {e}")
        return query

def patched_vector_cypher_search(self, query_text, top_k=5, **kwargs):
    # 1. KBPedia Search (via Neo4j)
    kb_items = []
    if hasattr(self, 'llm') and self.llm:
        if not hasattr(self, 'kbpedia_retriever'):
             from .kbpedia_retriever import KBPediaRetriever
             self.kbpedia_retriever = KBPediaRetriever(driver=self._driver, llm=self.llm, top_k=top_k)
        
        kb_result = self.kbpedia_retriever.search(query_text, top_k=top_k)
        if kb_result and kb_result.items:
            kb_items = kb_result.items

    # 2. Local Vector Search
    local_items = []
    MIN_SCORE = 0.4
    MIN_CONTENT_LENGTH = 20
    try:
        local_result = self._original_search(query_text=query_text, top_k=top_k, **kwargs)
        if local_result and local_result.items:
            filtered_items = []
            for item in local_result.items:
                # Cleaning
                if item.metadata and 'local_context' in item.metadata:
                    local_ctx = item.metadata['local_context']
                    if local_ctx and isinstance(local_ctx, list) and len(local_ctx) > 0:
                        clean_text = local_ctx[0].get('text')
                        if clean_text:
                            item.content = clean_text
                elif item.content and "<Record" in str(item.content):
                    match = re.search(r"text='(.*?)'", str(item.content))
                    if match:
                        item.content = match.group(1)

                score = item.metadata.get('score', 0.0) if item.metadata else 0.0
                content = (item.content or "").strip()
                if score >= MIN_SCORE and len(content) >= MIN_CONTENT_LENGTH:
                    filtered_items.append(item)
            local_items = filtered_items
    except Exception as e:
        print(f"DEBUG: Main retriever failed: {e}")
        
    all_items = kb_items + local_items
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
    
    final_top_k = top_k * 2 if (kb_items and local_items) else top_k
    return RetrieverResult(items=unique_items[:final_top_k], metadata={})


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
        r = HybridRetriever(
            driver=driver,
            vector_index_name="documentsVectorIndex",
            fulltext_index_name="documentsFulltextIndex",
            embedder=embedder,
            return_properties=["text", "id"] 
        )
        # Monkey-patch
        r._original_search = r.search
        r.search = types.MethodType(patched_hybrid_search, r)
        return r
        
    elif RETRIEVER.value == "HybridCypherRetriever":
        return HybridCypherRetriever(
            driver=driver,
            vector_index_name="documentsVectorIndex",
            fulltext_index_name="documentsFulltextIndex",
            retrieval_query=RETRIEVAL_QUERY,
            embedder=embedder
        )
    elif RETRIEVER.value == "VectorCypherRetriever":
        r = VectorCypherRetriever(
            driver=driver,
            index_name="documentsVectorIndex",
            retrieval_query=RETRIEVAL_QUERY,
            embedder=embedder
        )
        r._driver = driver  # Store driver ref for KBPedia Neo4j queries
        r.llm = None # Initialize llm attribute for patching
        # Monkey-patch
        r._original_search = r.search
        r.search = types.MethodType(patched_vector_cypher_search, r)
        return r
    else:
        raise ValueError(f"Retriever {RETRIEVER.name} is invalid.")
    
    # Monkey-patch driver if not present
    # This section is removed as it's no longer needed with the new patching approach.


def generate(llm, retriever, query) -> dict:
    # If the retriever is our EnhancedRetriever, inject the LLM
    if hasattr(retriever, 'llm'):
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
            # If item.content looks like "<Record ...>", we try to clean it.
            clean_content = item.content
            if clean_content and "<Record" in str(clean_content):
                # Try to extract text='...'
                match = re.search(r"text='(.*?)'", str(clean_content))
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
            
            # Check for knowledge graph triples (from KBPedia/VectorCypherRetriever)
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