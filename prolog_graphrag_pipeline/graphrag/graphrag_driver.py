import neo4j
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from .encoder import process_pdf_documents, process_text_context, process_context, extract_query_and_context
from .retriever import generate, create_retriever
import os
import glob
from .config import PROMPT_TEMPLATE, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, SCHEMA_CONFIG, STATIC_SCHEMA # added dot for relative import, assuming config.py is in the same directory as graphrag_driver.py
# However, this will fail when you run this file directly. For debugging purposes, you can run 'python -m graphrag-pipeline.graphrag.graphrag_driver' in the terminal.
import asyncio
import time
import sys

# Persistent event loop to avoid "RuntimeError: Event loop is closed" on Windows.
# asyncio.run() creates and destroys a new loop each call, but httpx/httpcore
# transport cleanup callbacks fire AFTER the loop is closed, causing a crash.
# By reusing a single loop, we keep it alive for the entire process lifetime.
_persistent_loop = None

def _run_async(coro):
    """Run an async coroutine using a persistent event loop (avoids httpx cleanup crash)."""
    global _persistent_loop
    if _persistent_loop is None or _persistent_loop.is_closed():
        _persistent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_persistent_loop)
    return _persistent_loop.run_until_complete(coro)

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
# llama3.1:8b-instruct uses 8GB VRAM, while q4_K_M quantized version uses around 4GB-5GB VRAM so it's the only one I can run locally
MODEL = "llama3.1:8b-instruct-q4_K_M"

class DebugOllamaLLM(OllamaLLM):
    """Wrapper to clean input prompts and log raw LLM responses to debug.txt."""
    
    def _log_to_file(self, message: str):
        """Appends a message to debug.txt with UTF-8 encoding."""
        try:
            with open("debug.txt", "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception as e:
            print(f"[DebugOllamaLLM Error] Could not write to debug.txt: {e}")

    def _clean_response_text(self, text: str) -> str:
        """Helper: Extracts JSON object from the response string.

        Always prefers {…} objects because SimpleKGPipeline strictly requires
        {"nodes": […], "relationships": […]}. The triple filter (filter_triples_for_query)
        does its own independent [bracket] extraction and does NOT rely on this method
        to return arrays, so reverting to object-only is safe for both callers.
        """
        if not text:
            return ""
        obj_start = text.find('{')
        obj_end   = text.rfind('}')
        if obj_start != -1 and obj_end != -1 and obj_start < obj_end:
            return text[obj_start : obj_end + 1]
        return text   # nothing to strip — return as-is


    def _clean_prompt(self, text: str) -> str:
        """Removes all newlines from the input prompt to fix PDF formatting."""
        return text.replace('\n', ' ')

    def invoke(self, input: str, **kwargs) -> str:
        # 1. Clean the input
        clean_input = self._clean_prompt(input)
        
        # 2. LOG to file instead of print
        self._log_to_file(f"\n--- [DEBUG] SENT PROMPT (Sync) ---\n{clean_input}\n----------------------------------")

        print(f"  > Sending chunk to LLM ({len(input)} chars)...", end="", flush=True)
        
        # 3. Send to LLM
        response = super().invoke(clean_input, **kwargs)

        print(" Done!")
        
        # 4. Handle response
        raw_content = response.content if hasattr(response, 'content') else str(response)
        
        # 5. LOG to file
        self._log_to_file(f"\n=== [DEBUG] RAW RESPONSE (Sync) ===\n{raw_content}\n===================================")
        
        return self._clean_response_text(raw_content)

    async def ainvoke(self, input: str, **kwargs):
        # 1. Clean the input
        clean_input = self._clean_prompt(input)
        
        # 2. LOG to file
        self._log_to_file(f"\n--- [DEBUG] SENT PROMPT (Async) ---\n{clean_input}\n----------------------------------")
        
        print(f"  > [Async] Sending chunk to LLM ({len(input)} chars) and waiting for extraction...", end="", flush=True)
        
        # 3. Send to LLM
        response = await super().ainvoke(clean_input, **kwargs)
        
        print(" Done!")
        
        # 4. Extract content
        raw_content = response.content if hasattr(response, 'content') else str(response)
        
        # 5. LOG to file
        self._log_to_file(f"\n=== [DEBUG] RAW RESPONSE (Async) ===\n{raw_content}\n===================================")
        
        clean_content = self._clean_response_text(raw_content)
        
        # 6. Return Object or String
        if hasattr(response, 'content'):
            response.content = clean_content
            return response
        else:
            return clean_content
        

neo4j_driver = None


def initialize_models() -> tuple[OllamaLLM, OllamaLLM, OllamaEmbeddings]:
    # keep_alive=0 unloads VRAM after each document processing
    encoder_llm = DebugOllamaLLM(
        model_name=MODEL,
        model_params={
            "keep_alive": "20m",
            "options": { # in case parameters need to be nested
                "temperature": 0,
                "num_ctx": 2048,
                "format": "json",
                "repeat_penalty": 1.2,
                "seed": 42 # for reproducibility
            }
        }
    )

    # 2. The Retriever: Logical synthesis for Prolog
    retriever_llm = OllamaLLM(
        model_name=MODEL,
        model_params={
            "keep_alive": "20m", # Keep it loaded during an active chat session
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048,
                "format": "json",
                "repeat_penalty": 1.2,
                "seed": 42 # for reproducibility
            }
        
        }
    )

    # 2. Initialize Embedder
    embedder = OllamaEmbeddings(
        model=MODEL
    )

    print("Ollama loaded successfully.")
    
    return encoder_llm, retriever_llm, embedder

def setup_kg_pipeline(llm, embedder) -> SimpleKGPipeline    :
    STATIC_SCHEMA = { # NOTE: Due to how 'EXTRACTED' works, the prompt used by SimpleKGPipeline is bad and does not constrain the output to only JSON format. We can create a custom KG Pipeline but for now we use this to skip it entirely.
        "node_types": [
            "Concept",      # The heavy lifter: covers Theorems, Ideas, Theories, Terms
            "Person",       # Key figures: Socrates, Einstein, Historical Leaders
            "Event",        # Occurrences: Wars, Experiments, Historical Periods
            "Definition",   # Crucial for education: Explicit statements of meaning
            "Example",      # Grounding: Specific instances (e.g., "Pingu" or "3")
            "Topic"         # The Field: "Mathematics", "Philosophy", "Biology"
        ],
        "relationship_types": [
            "RELATED_TO",   # Generic connection
            "IS_A",         # Hierarchy: "Calculus IS_A Topic", "Penguin IS_A Bird"
            "PART_OF",      # Composition: "Cell PART_OF Tissue"
            "CAUSES",       # Logic/History: "War CAUSES Famine", "Premise CAUSES Conclusion"
            "DEFINES",      # Concept -> Definition
            "ILLUSTRATES"   # Example -> Concept ("Pingu ILLUSTRATES Non-flying bird")
        ]
    }
    chunk_splitter = FixedSizeSplitter(chunk_size=800, chunk_overlap=100) # OPT2: reduced from 1200/150 → fewer tokens per extraction call
    
    schema = STATIC_SCHEMA if SCHEMA_CONFIG.name == "STATIC" else "EXTRACTED"

    kg_builder_pdf = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=embedder,
        entities=schema["node_types"],         
        relations=schema["relationship_types"],  
        prompt_template=PROMPT_TEMPLATE,
        from_pdf=True,
        text_splitter=chunk_splitter,
        perform_entity_resolution=True,
        on_error="IGNORE"
    )
    
    kg_builder_text = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=embedder,
        entities=schema["node_types"],         
        relations=schema["relationship_types"],  
        prompt_template=PROMPT_TEMPLATE,
        from_pdf=False,
        text_splitter=chunk_splitter,
        perform_entity_resolution=True,
        on_error="IGNORE"
    )
    
    return kg_builder_pdf, kg_builder_text

encoder_llm = None
retriever_llm = None
embedder = None
kg_builder_pdf = None
kg_builder_text = None
retriever = None

def init_globals():
    global encoder_llm, retriever_llm, embedder, kg_builder_pdf, kg_builder_text, retriever, neo4j_driver
    
    driver_recreated = ensure_driver_connected()

    if encoder_llm is None or driver_recreated:
        print("DEBUG: Re-initializing pipelines due to missing LLM or new driver...", flush=True)
        encoder_llm, retriever_llm, embedder = initialize_models()
        kg_builder_pdf, kg_builder_text = setup_kg_pipeline(encoder_llm, embedder)
        retriever = create_retriever(neo4j_driver, embedder)

def generate_answer(query, retriever, llm, original_query: str = "") -> dict:
    return generate(llm, retriever, query, original_query=original_query)

def run_pipeline(user_question) -> dict:
    init_globals()
    
    # 1. Clear stale Neo4j data from previous questions to prevent irrelevant retrieval
    print("DEBUG [run_pipeline]: Clearing local data...", flush=True)
    clear_local_data()
    print("DEBUG [run_pipeline]: Local data cleared. Starting extract_query_and_context...", flush=True)
    
    # 2. Extract query and context from user input
    query, text_context = extract_query_and_context(user_question)
    print(f"DEBUG [run_pipeline]: extract_query_and_context done. query={repr(query[:80])} | context items={len(text_context) if isinstance(text_context, list) else 'str'}", flush=True)
    
    # 3. OPT3: Only ingest into the KG if the encoder actually found real context
    #    (e.g. a passage-based question). For self-contained MCQs, context is empty —
    #    ingesting the question itself produces barely-useful local chunks (~0.7 score)
    #    while adding a full LLM call + 2s sleep per question. Skip it.
    context_was_extracted = bool(text_context)  # True only if encoder found real facts/passages
    
    if not context_was_extracted:
        print("DEBUG [run_pipeline]: No extracted context (MCQ). Skipping KG ingestion — querying KBPedia directly.", flush=True)
        text_context = [user_question]  # keep for logging/reference, but don't ingest
    else:
        print(f"DEBUG [run_pipeline]: Starting process_context (KG ingestion) with {len(text_context)} text chunk(s)...", flush=True)
        _run_async(process_context(kg_builder_pdf=kg_builder_pdf, kg_builder_text=kg_builder_text, texts=text_context))
        print("DEBUG [run_pipeline]: process_context done. Sleeping 2s for index update...", flush=True)
        time.sleep(2) # Allow vector index to update
        print("DEBUG [run_pipeline]: Sleep done.", flush=True)

    print("DEBUG [run_pipeline]: Starting generate (retriever + LLM)...", flush=True)
    generation_result = generate(llm=retriever_llm, retriever=retriever, query=query, original_query=user_question)
    print("DEBUG [run_pipeline]: generate done. Returning result.", flush=True)
    # generation_result is now a dict: {'answer': str, 'contexts': list}
    
    return {
        "query": query,
        "text_context": text_context,
        "answer": generation_result["answer"],
        "contexts": generation_result["contexts"]
            }

async def test_process_pdf_context(kg_builder_pdf: SimpleKGPipeline):
    await process_pdf_documents(kg_builder_pdf)
    
async def test_process_text_context(kg_builder_text: SimpleKGPipeline, query: str):
    await process_text_context(kg_builder_text)
    
def test_extract_query_and_context(user_prompt: str):
    query, context = extract_query_and_context(user_prompt)
    print(f"Extracted query: {query}")
    print(f"Extracted context: {context}")
    return query, context
    
async def perform_tests():
    # 1. Initialize models and pipelines
    llm, retriever_llm, embedder = initialize_models()
    kg_builder_pdf, kg_builder_text = setup_kg_pipeline(llm, embedder)
    
    # 2. Extract Phase (Synchronous)
    print("--- Testing extract_query_and_context ---")
    question = "What object found on the solid glass table was able to unlock the fifteen-inch door?"
    # "Which of the following sets is defined by the Well-Ordering Principle as having a guaranteed smallest element? A) Empty sets B) Sets of negative integers C) Nonempty sets of nonnegative integers D) Sets of positive rationals"
    
    # Corrected: Removed the extra comma that was making this a tuple
    query, context = test_extract_query_and_context(question)

    # 3. Processing Phase (Asynchronous)
    # Corrected: Do NOT use 'await' INSIDE asyncio.gather. 
    # Pass the coroutines themselves to gather.
    print("\n--- Starting Knowledge Graph Ingestion ---")
    
    await asyncio.gather(
        test_process_pdf_context(kg_builder_pdf),
        # process_text_context(kg_builder_text, context)
    )
    
    print("\n--- All tests completed ---")


def ensure_driver_connected():
    global neo4j_driver
    while True:
        # print("DEBUG: Checking Neo4j driver...", flush=True)
        try:
            if neo4j_driver is not None:
                # print("DEBUG: Verifying existing Neo4j connectivity...", flush=True)
                neo4j_driver.verify_connectivity()
                # print("DEBUG: Neo4j connectivity verified.", flush=True)
                return False

            print("DEBUG: Initializing new Neo4j driver...", flush=True)
            neo4j_driver = neo4j.GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
                connection_timeout=10.0, 
                max_connection_lifetime=200, 
                encrypted=False
            )
            neo4j_driver.verify_connectivity()
            print("DEBUG: Neo4j driver initialized and connected.", flush=True)
            return True
        except Exception as e:
            print(f"CRITICAL ERROR: Could not connect to Neo4j: {e}", flush=True)
            if neo4j_driver:
                try:
                    neo4j_driver.close()
                except:
                    pass
            neo4j_driver = None 
            
            print(f"Error: {e}")
            print("The process is paused. Please ensure Neo4j is running.")
            user_response = input("Press Enter to retry connecting... (or type 'q' to raise error): ")
            if user_response.lower().strip() == 'q':
                raise e
            print("Retrying connection...", flush=True)


def clear_local_data():
    """
    Deletes pipeline-generated nodes/relationships from the database,
    but preserves KBPedia reference concepts which are a static knowledge base.
    """
    ensure_driver_connected()
    print("WARNING: Clearing pipeline data (preserving KBPedia)...")
    try:
        with neo4j_driver.session(database="neo4j") as session:
            session.run("MATCH (n) WHERE NOT n:KBPediaConcept DETACH DELETE n")
    except Exception as e:
        print(f"Error clearing local data: {e}")
    print("Database cleared successfully (KBPedia preserved).")

def main():
    # Uncomment to clear data before running tests
    # clear_local_data() 
    _run_async(perform_tests())


if __name__ == "__main__":
    main()
    
# def generate_context(query, retriever, llm) -> str:
#     return generate(llm, retriever, query)