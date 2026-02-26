import os
import sys
import time
from pathlib import Path

# Add thesis root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import graphrag_pipeline.graphrag.graphrag_driver as gd
from prolog_graphrag_pipeline.graphrag.kbpedia_retriever import KBPediaRetriever

def test_extraction_methods():
    print("Initializing LLM and Driver...")
    gd.ensure_driver_connected()
    encoder_llm, retriever_llm, embedder = gd.initialize_models()
    
    retriever = KBPediaRetriever(driver=gd.neo4j_driver, llm=retriever_llm, top_k=5)

    # Test cases covering different scenarios
    test_cases = [
        {
            "name": "1. Exact Match (Entities are explicitly in the string)",
            "query": "What is the function of the cell membrane and cell wall in a plant cell?"
        },
        {
            "name": "2. Exact Match with MCQ Choices",
            "query": "Which of the following describes the fallacy of appeal to common practice? A. eliminating a source of evidence from consideration B. arguing that something is the right thing to do based only on evidence that other people are already doing it."
        },
        {
            "name": "3. Inferred Match (Requires understanding the underlying concept not explicitly named)",
            "query": "If I drop a hammer and a feather on the moon, why do they hit the ground at the same time? A. lack of atmospheric drag B. equal mass C. strong magnetic field"
        },
        {
            "name": "4. Complex Biological Pathway (Requires expanding on functional terms)",
            "query": "Which process is responsible for the conversion of light energy into chemical energy in plants, ultimately producing glucose and oxygen?"
        }
    ]

    for tc in test_cases:
        print("\n" + "="*80)
        print(f"SCENARIO: {tc['name']}")
        print(f"QUERY: {tc['query']}")
        print("="*80)

        # --- Fast Extraction ---
        print("\n>>> Running FAST Extraction (Keyword-based)")
        t0 = time.time()
        fast_entities = retriever._extract_entities_fast(tc['query'])
        t_fast = time.time() - t0
        print(f"Time: {t_fast:.2f}s")
        print(f"Entities: {fast_entities}")

        # --- LLM Extraction ---
        print("\n>>> Running LLM Extraction (extract_entities_from_query)")
        t0 = time.time()
        llm_entities = retriever.extract_entities_from_query(tc['query'])
        t_llm = time.time() - t0
        print(f"Time: {t_llm:.2f}s")
        print(f"Entities: {llm_entities}")

        print("\n" + "-"*40)
        print(f"Speed comparison: Fast is {t_llm - t_fast:.2f}s faster than LLM.")

        # Simulate how GraphRAG would handle these entities by finding concepts
        print("\n>>> Concept Matching Simulation (Top 2 matches per top 3 entities):")
        
        print("Fast Concepts:")
        for entity in fast_entities[:3]:
            matches = retriever.find_concepts(entity, limit=2)
            names = [m['name'] for m in matches]
            print(f"  - '{entity}' -> {names}")
            
        print("\nLLM Concepts:")
        for entity in llm_entities[:3]:
            matches = retriever.find_concepts(entity, limit=2)
            names = [m['name'] for m in matches]
            print(f"  - '{entity}' -> {names}")

if __name__ == "__main__":
    test_extraction_methods()
