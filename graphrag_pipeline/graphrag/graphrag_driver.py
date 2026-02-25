import neo4j
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
import os
import glob
from .config import PROMPT_TEMPLATE, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, SCHEMA_CONFIG, STATIC_SCHEMA # added dot for relative import, assuming config.py is in the same directory as graphrag_driver.py
# However, this will fail when you run this file directly. For debugging purposes, you can run 'python -m graphrag-pipeline.graphrag.graphrag_driver' in the terminal.
import asyncio

import sys
from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parents[1]))
# llama3.1:8b-instruct uses 8GB VRAM, while q4_K_M quantized version uses around 4GB-5GB VRAM so it's the only one I can run locally
MODEL = "llama3.1:8b-instruct-q4_K_M"

try:
    from graphrag_pipeline.llm import generate
except ModuleNotFoundError:
    # Fallback for direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from llm import generate

class DebugOllamaLLM(OllamaLLM):
    """Wrapper to clean input prompts and log raw LLM responses to debug.txt."""
    
    def _log_to_file(self, message: str):
        """Appends a message to debug.txt with UTF-8 encoding."""
        try:
            # "a" mode appends to the file. Change to "w" if you want to overwrite every time.
            with open("debug.txt", "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception as e:
            # Fallback to console if file writing fails
            print(f"[DebugOllamaLLM Error] Could not write to debug.txt: {e}")

    def _clean_response_text(self, text: str) -> str:
        """Helper: Extracts JSON from the response string."""
        if not text:
            return ""
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            return text[start_index : end_index + 1]
        return text

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
        
        # 3. Send to LLM
        response = await super().ainvoke(clean_input, **kwargs)
        
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
            # raise Exception("Mock Connection Failure")
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
            "keep_alive": "5m", # Keep it loaded during an active chat session
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
    
    chunk_splitter = FixedSizeSplitter(chunk_size=1200, chunk_overlap=150) # changed from 2000 and 200 to smaller sizes for better context handling
    
    schema = STATIC_SCHEMA if SCHEMA_CONFIG.name == "STATIC" else "EXTRACTED"

    kg_builder_pdf = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=embedder,
        schema=schema, 
        prompt_template=PROMPT_TEMPLATE,
        from_pdf=True,
        text_splitter=chunk_splitter,
        on_error="IGNORE"
    )
    
    kg_builder_text = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=embedder,
        schema=schema,
        prompt_template=PROMPT_TEMPLATE,
        from_pdf=False,
        text_splitter=chunk_splitter,
        on_error="IGNORE"
    )
    
    return kg_builder_pdf, kg_builder_text

def generate_answer(query, retriever, llm, original_query: str = "") -> dict:
    # Inject LLM into EnhancedRetriever so KBPedia search is active
    from .retriever import EnhancedRetriever
    if isinstance(retriever, EnhancedRetriever):
        retriever.llm = llm
    return generate(llm, retriever, query, original_query=original_query)

from .encoder import process_pdf_documents, process_text_context, extract_query_and_context

async def test_process_pdf_context(kg_builder_pdf: SimpleKGPipeline):
    await process_pdf_documents(kg_builder_pdf)
    
    
async def perform_tests():
    # 1. Initialize models and pipelines
    ensure_driver_connected() # Added ensures connected
    llm, retriever_llm, embedder = initialize_models() # forgot retriever_llm here, but it's not used in the tests below, so it's fine for now
    kg_builder_pdf, kg_builder_text = setup_kg_pipeline(llm, embedder)
    
    # 2. Extract Phase (Synchronous)
    print("--- Testing extract_query_and_context ---")
    question = "What object found on the solid glass table was able to unlock the fifteen-inch door?"
    # "Which of the following sets is defined by the Well-Ordering Principle as having a guaranteed smallest element? A) Empty sets B) Sets of negative integers C) Nonempty sets of nonnegative integers D) Sets of positive rationals"
    
    # Corrected: Removed the extra comma that was making this a tuple

    # 3. Processing Phase (Asynchronous)
    # Corrected: Do NOT use 'await' INSIDE asyncio.gather. 
    # Pass the coroutines themselves to gather.
    print("\n--- Starting Knowledge Graph Ingestion ---")
    
    await asyncio.gather(
        test_process_pdf_context(kg_builder_pdf),
        # process_text_context(kg_builder_text, context)
    )
    
    print("\n--- All tests completed ---")

def main():
    asyncio.run(perform_tests())

if __name__ == "__main__":
    main()
    
# def generate_context(query, retriever, llm) -> str:
#     return generate(llm, retriever, query)

from .retriever import get_retriever, create_indexes, wait_for_indexes

def run_pipeline(question: str) -> dict:
    """
    Runs the GraphRAG pipeline for a given question.
    Returns a dictionary with 'answer' and 'contexts'.
    Retriever uses KBPedia (global ontology) + local vector search.
    """
    try:
        ensure_driver_connected()

        # 1. Initialize models
        encoder_llm, retriever_llm, embedder = initialize_models()

        # 2. Ensure indexes exist (self-healing for new Neo4j instances)
        print("DEBUG [graphrag run_pipeline]: Verifying Neo4j indexes...", flush=True)
        create_indexes(neo4j_driver)
        wait_for_indexes(neo4j_driver, timeout=30)

        # 3. Get retriever and inject LLM so KBPedia search is active
        retriever = get_retriever(neo4j_driver, embedder)
        if hasattr(retriever, 'llm'):
            retriever.llm = retriever_llm

        # 4. Generate answer — pass full question as original_query for MCQ-aware KBPedia filter
        result = generate_answer(question, retriever, retriever_llm, original_query=question)

        return result
    except Exception as e:
        print(f"Error in run_pipeline: {e}")
        return {"answer": "Error", "contexts": []}