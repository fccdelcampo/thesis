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
        

neo4j_driver = neo4j.GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )


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
    chunk_splitter = FixedSizeSplitter(chunk_size=1200, chunk_overlap=150) # changed from 2000 and 200 to smaller sizes for better context handling
    
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
    global encoder_llm, retriever_llm, embedder, kg_builder_pdf, kg_builder_text, retriever
    if encoder_llm is None:
        encoder_llm, retriever_llm, embedder = initialize_models()
        kg_builder_pdf, kg_builder_text = setup_kg_pipeline(encoder_llm, embedder)
        retriever = create_retriever(neo4j_driver, embedder)

def generate_answer(query, retriever, llm) -> dict:
    return generate(llm, retriever, query)

def run_pipeline(user_question) -> dict:
    init_globals()
    query, text_context = extract_query_and_context(user_question)
    asyncio.run(process_context(kg_builder_pdf=kg_builder_pdf, kg_builder_text=kg_builder_text, texts=text_context))
    time.sleep(2) # Allow vector index to update
    generation_result = generate(llm=retriever_llm, retriever=retriever, query=query)
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
    llm, retriever_llm, embedder = initialize_models() # forgot retriever_llm here, but it's not used in the tests below, so it's fine for now
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

def clear_local_data():
    """
    Deletes all Local Document and Chunk nodes from the database.
    """
    print("WARNING: Deleting all Local Documents and Chunks...")
    with neo4j_driver.session() as session:
        session.run("MATCH (d:Document) DETACH DELETE d")
        session.run("MATCH (c:Chunk) DETACH DELETE c")
    print("Local data cleared successfully.")

def main():
    # Uncomment to clear data before running tests
    # clear_local_data() 
    asyncio.run(perform_tests())


if __name__ == "__main__":
    main()
    
# def generate_context(query, retriever, llm) -> str:
#     return generate(llm, retriever, query)