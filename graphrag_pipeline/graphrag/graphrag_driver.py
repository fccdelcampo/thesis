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

# Ensure project root is on the path so llm_config can be imported
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from llm_config import MODEL_NAME as MODEL, USE_TOGETHER_API, get_openai_client, BASE_URL, API_KEY

if USE_TOGETHER_API:
    from neo4j_graphrag.embeddings.base import Embedder

    class TogetherAIEmbeddings(Embedder):
        """Embedder that calls the Together AI OpenAI-compatible embeddings endpoint."""
        def embed_query(self, text: str):
            from llm_config import EMBED_MODEL
            _client = get_openai_client()
            response = _client.embeddings.create(
                model=EMBED_MODEL,
                input=text,
            )
            return response.data[0].embedding

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
        # Strip markdown code fences first (```json ... ``` or ``` ... ```)
        import re
        text = re.sub(r'```(?:json)?\s*', '', text).strip()
        obj_start = text.find('{')
        obj_end   = text.rfind('}')
        if obj_start != -1 and obj_end != -1 and obj_start < obj_end:
            return text[obj_start : obj_end + 1]
        return text  # Not JSON — return as-is (e.g. plain QA answer)

    def _clean_prompt(self, text: str) -> str:
        """Removes newlines only from PDF-extracted prose.

        Structured filter prompts (KBPedia batch filter, concept filter) use
        intentional newlines to separate concept blocks. Stripping them collapses
        all structure into one line, which confuses the model and causes hangs.
        We detect structured prompts by the presence of known marker strings.
        """
        STRUCTURED_MARKERS = ("Concept: '", "  - ", "Facts:\n", "Candidates (")
        if any(m in text for m in STRUCTURED_MARKERS):
            return text   # preserve structured prompt formatting
        return text.replace('\n', ' ')

    def _clean_input_data(self, input_data):
        """Applies _clean_prompt either to a string or to the 'content' of a list of message dicts."""
        if isinstance(input_data, str):
            return self._clean_prompt(input_data)
        elif isinstance(input_data, list):
            # Neo4j GraphRAG passes List[LLMMessage] which are dicts with 'role' and 'content'
            cleaned_list = []
            for msg in input_data:
                cleaned_msg = msg.copy() if isinstance(msg, dict) else msg
                if isinstance(cleaned_msg, dict) and 'content' in cleaned_msg:
                    cleaned_msg['content'] = self._clean_prompt(str(cleaned_msg['content']))
                cleaned_list.append(cleaned_msg)
            return cleaned_list
        return input_data

    def _openai_invoke(self, input, message_history=None, system_instruction=None):
        """Route through the openai client when USE_TOGETHER_API=True."""
        from neo4j_graphrag.llm.types import LLMResponse
        _client = get_openai_client()
        if isinstance(input, list):
            messages = input
        else:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": str(system_instruction)})
            if message_history:
                messages.extend([{"role": m.get("role", "user"), "content": m.get("content", "")} for m in message_history])
            messages.append({"role": "user", "content": str(input)})
        response = _client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        return LLMResponse(content=self._clean_response_text(content))

    def invoke(self, input, message_history=None, system_instruction=None, **kwargs):
        if USE_TOGETHER_API:
            return self._openai_invoke(input, message_history, system_instruction)
        kwargs.pop('response_format', None) # neo4j_graphrag OllamaLLM hates response_format
        clean_input = self._clean_input_data(input)
        
        # Format for logging
        log_input = clean_input if isinstance(clean_input, str) else str(clean_input)
        self._log_to_file(f"\n--- [DEBUG] SENT PROMPT (Sync) ---\n{log_input}\n----------------------------------")
        
        input_len = len(input) if isinstance(input, str) else len(str(input))
        print(f"  > Sending chunk to LLM ({input_len} chars)...", end="", flush=True)
        
        response = super().invoke(
            input=clean_input,
            message_history=message_history,
            system_instruction=system_instruction
        )
        print(" Done!")
        
        raw_content = response.content if hasattr(response, 'content') else str(response)
        self._log_to_file(f"\n=== [DEBUG] RAW RESPONSE (Sync) ===\n{raw_content}\n===================================")
        
        clean_content = self._clean_response_text(raw_content)
        if hasattr(response, 'content'):
            response.content = clean_content
            return response
        else:
            from neo4j_graphrag.llm.types import LLMResponse
            return LLMResponse(content=clean_content)

    async def ainvoke(self, input, message_history=None, system_instruction=None, **kwargs):
        if USE_TOGETHER_API:
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._openai_invoke, input, message_history, system_instruction)
        kwargs.pop('response_format', None)
        clean_input = self._clean_input_data(input)
        
        log_input = clean_input if isinstance(clean_input, str) else str(clean_input)
        self._log_to_file(f"\n--- [DEBUG] SENT PROMPT (Async) ---\n{log_input}\n----------------------------------")
        
        input_len = len(input) if isinstance(input, str) else len(str(input))
        print(f"  > [Async] Sending chunk to LLM ({input_len} chars) and waiting for extraction...", end="", flush=True)
        
        # Bypass neo4j-graphrag's ainvoke and its broken rate limit decorator
        # Call the native async_client from ollama directly
        from neo4j_graphrag.llm.types import LLMResponse
        try:
            if isinstance(clean_input, str):
                messages = self.get_messages(clean_input, message_history=message_history, system_instruction=system_instruction)
            else:
                messages = self.get_messages_v2(clean_input)
                
            response_obj = await self.async_client.chat(
                model=self.model_name,
                messages=messages,
                options={**self.model_params, **kwargs},
            )
            raw_content = response_obj.message.content or ""
            response = LLMResponse(content=raw_content)
        except Exception as e:
            print(f"\n[DebugOllamaLLM Error] Async call failed: {e}")
            raise
        
        print(" Done!")
        
        self._log_to_file(f"\n=== [DEBUG] RAW RESPONSE (Async) ===\n{raw_content}\n===================================")
        
        clean_content = self._clean_response_text(raw_content)
        
        if hasattr(response, 'content'):
            response.content = clean_content
            return response
        else:
            return clean_content
        


neo4j_driver = None

def ensure_driver_connected():
    global neo4j_driver
    MAX_RETRIES = 10
    RETRY_DELAY = 30  # seconds between retries
    
    for attempt in range(MAX_RETRIES):
        try:
            if neo4j_driver is not None:
                neo4j_driver.verify_connectivity()
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
            print(f"CRITICAL ERROR: Could not connect to Neo4j (attempt {attempt+1}/{MAX_RETRIES}): {e}", flush=True)
            if neo4j_driver:
                try:
                    neo4j_driver.close()
                except:
                    pass
            neo4j_driver = None
            
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY}s...", flush=True)
                import time
                time.sleep(RETRY_DELAY)
            else:
                raise ConnectionError(f"Failed to connect to Neo4j after {MAX_RETRIES} attempts: {e}") from e

def initialize_models():
    encoder_llm = DebugOllamaLLM(
        model_name=MODEL,
        model_params={
            "keep_alive": "20m",
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "format": "json",
                "repeat_penalty": 1.2,
                "seed": 42
            }
        }
    )

    retriever_llm = DebugOllamaLLM(
        model_name=MODEL,
        model_params={
            "keep_alive": "5m",
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096,
                "repeat_penalty": 1.2,
                "seed": 42
            }
        }
    )

    if USE_TOGETHER_API:
        from llm_config import EMBED_MODEL
        embedder = TogetherAIEmbeddings()
        print(f"Together AI embedder loaded ({EMBED_MODEL}).")
    else:
        from llm_config import EMBED_MODEL
        embedder = OllamaEmbeddings(model=EMBED_MODEL)
        print("Ollama embedder loaded.")

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


def clear_local_data():
    """
    Deletes pipeline-generated nodes/relationships, preserving KBPedia reference concepts.
    Called at the start of every run to flush any stale data from interrupted runs.
    """
    try:
        with neo4j_driver.session(database="neo4j") as session:
            session.run("MATCH (n) WHERE NOT n:KBPediaConcept DETACH DELETE n")
        print("DEBUG: Stale local documents cleared (KBPedia preserved).", flush=True)
    except Exception as e:
        print(f"DEBUG: clear_local_data failed (non-fatal): {e}", flush=True)


def run_pipeline(question: str) -> dict:
    """
    Runs the GraphRAG pipeline for a given question.
    Returns a dictionary with 'answer' and 'contexts'.
    Retriever uses KBPedia (global ontology) + local vector search.
    """
    try:
        ensure_driver_connected()

        # 0. Clear any stale local documents from a previous (possibly interrupted) run
        clear_local_data()

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