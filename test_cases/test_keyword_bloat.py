import os
import sys
import time
from pathlib import Path

# Add thesis root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import graphrag_pipeline.graphrag.graphrag_driver as gd
from prolog_graphrag_pipeline.graphrag.kbpedia_retriever import KBPediaRetriever

def test_keyword_bloat():
    print("Initializing LLM and Driver...")
    gd.ensure_driver_connected()
    encoder_llm, retriever_llm, embedder = gd.initialize_models()
    
    retriever_standard = KBPediaRetriever(driver=gd.neo4j_driver, llm=retriever_llm, top_k=5)
    retriever_bloated = KBPediaRetriever(driver=gd.neo4j_driver, llm=retriever_llm, top_k=5)

    test_cases = [
        {
            "name": "1. Standard Benchmark (Moderate complexity)",
            "query": "Which of the following describes the fallacy of appeal to common practice? A. eliminating a source of evidence from consideration B. arguing that something is the right thing to do based only on evidence that other people are already doing it."
        },
        {
            "name": "2. Inferred Knowledge (Abstract Physics)",
            "query": "If I drop a hammer and a feather on the moon, why do they hit the ground at the same time? A. lack of atmospheric drag B. equal mass C. strong magnetic field"
        },
        {
            "name": "3. Complex Biology (Heavy jargon)",
            "query": "Which process is responsible for the conversion of light energy into chemical energy in plants, ultimately producing glucose and oxygen?"
        }
    ]

    for tc in test_cases:
        print("\n" + "="*80)
        print(f"SCENARIO: {tc['name']}")
        print(f"QUERY: {tc['query']}")
        print("="*80)

        base_query = tc['query'].split('?')[0] + '?'

        # --- 1. Standard Hybrid Retrieval ---
        print("\n>>> Running STANDARD Hybrid Retrieval (MCQ + LLM Conceptual)")
        t0 = time.time()
        res_standard = retriever_standard.search(query_text=base_query, top_k=5, original_query=tc['query'], include_keywords=False)
        t_standard = time.time() - t0
        
        entities_standard = retriever_standard.extract_entities(base_query, include_keywords=False)

        # --- 2. Bloated Hybrid Retrieval (MCQ + LLM + All Keywords) ---
        print("\n>>> Running BLOATED Hybrid Retrieval (MCQ + LLM + Keywords)")
        t0 = time.time()
        res_bloat = retriever_bloated.search(query_text=base_query, top_k=5, original_query=tc['query'], include_keywords=True)
        t_bloat = time.time() - t0
        
        entities_bloat = retriever_bloated.extract_entities(base_query, include_keywords=True)

        # --- SUMMARY ---
        print("\n" + "-"*40)
        print(f"RESULTS FOR: {tc['name']}")
        print(f"Entities (Standard) [{len(entities_standard)}]: {entities_standard}")
        print(f"Entities (Bloated)  [{len(entities_bloat)}]: {entities_bloat}")
        
        print(f"\nTime (Standard): {t_standard:.2f}s")
        print(f"Time (Bloated) : {t_bloat:.2f}s  <-- {t_bloat - t_standard:+.2f}s Overhead")

        print(f"\nFinal Triples Kept (Standard): {sum(len(item.metadata.get('triples', [])) for item in res_standard.items)}")
        print(f"Final Triples Kept (Bloated) : {sum(len(item.metadata.get('triples', [])) for item in res_bloat.items)}")

        # Check concept overlap
        concepts_std = set([item.metadata['entity'] for item in res_standard.items])
        concepts_blt = set([item.metadata['entity'] for item in res_bloat.items])
        
        only_std = concepts_std - concepts_blt
        only_blt = concepts_blt - concepts_std
        
        print("\nConcept Differences:")
        if not only_std and not only_blt:
            print("  [Exactly the same concepts kept!]")
        else:
            if only_std:
                print(f"  Only in Standard: {only_std}")
            if only_blt:
                print(f"  Only in Bloated: {only_blt}")

if __name__ == "__main__":
    test_keyword_bloat()
