import os
import sys
import time
from pathlib import Path

# Add thesis root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import graphrag_pipeline.graphrag.graphrag_driver as gd
from prolog_graphrag_pipeline.graphrag.kbpedia_retriever import KBPediaRetriever

def print_result_comparison(query: str, seq_res, batch_res):
    print("\n" + "#"*80)
    print(f"QUERY: {query}")
    print("#"*80)

    seq_items = seq_res.items
    batch_items = batch_res.items

    print(f"\n[Concepts Kept] Sequential: {len(seq_items)} | Batch: {len(batch_items)}")

    # Extract concept -> triples mapping
    seq_map = {item.metadata['entity']: item.metadata['triples'] for item in seq_items}
    batch_map = {item.metadata['entity']: item.metadata['triples'] for item in batch_items}

    all_concepts = set(seq_map.keys()).union(set(batch_map.keys()))

    for concept in all_concepts:
        print(f"\n--- Concept: {concept} ---")
        seq_triples = seq_map.get(concept, [])
        batch_triples = batch_map.get(concept, [])

        print(f"  Sequential ({len(seq_triples)} facts kept):")
        for t in seq_triples:
            print(f"    - {t}")

        print(f"\n  Batch ({len(batch_triples)} facts kept):")
        for t in batch_triples:
            print(f"    - {t}")
            
        # Highlight differences
        seq_set = set(seq_triples)
        batch_set = set(batch_triples)
        
        only_seq = seq_set - batch_set
        only_batch = batch_set - seq_set
        
        if only_seq or only_batch:
            print(f"\n  [!] DIFFERENCES DETECTED for '{concept}':")
            if only_seq:
                print("      V-- Only kept by SEQUENTIAL --V")
                for t in only_seq:
                    print(f"      - {t}")
            if only_batch:
                print("      V-- Only kept by BATCH --V")
                for t in only_batch:
                    print(f"      - {t}")
        else:
            print(f"\n  [==== EXACT MATCH ====]")

def test_filter_accuracy():
    print("Initializing LLM and Driver...")
    gd.ensure_driver_connected()
    encoder_llm, retriever_llm, embedder = gd.initialize_models()
    
    # 1. Setup Retrievers
    print("\nInitializing Sequential Retriever...")
    retriever_seq = KBPediaRetriever(driver=gd.neo4j_driver, llm=retriever_llm, top_k=5)
    # Patch the retriever to use sequential filtering instead of batch
    original_batch = retriever_seq.filter_triples_batch
    def force_sequential(query, batch_data, original_query):
        res = {}
        for c in batch_data:
            res[c['name']] = retriever_seq.filter_triples_for_query(query, c['name'], c['triples'], original_query)
        return res
    retriever_seq.filter_triples_batch = force_sequential

    print("Initializing Batch Retriever...")
    retriever_batch = KBPediaRetriever(driver=gd.neo4j_driver, llm=retriever_llm, top_k=5)

    # 2. Define Test Cases (ranging from simple to complex with multiple concepts/triples)
    queries = [
        # Test Case 1: Standard AI2 ARC Question (Moderate Triples)
        "Which of the following describes the fallacy of appeal to common practice? A. eliminating a source of evidence from consideration by claiming the source is flawed B. arguing that something is the right thing to do based only on evidence that other people are already doing it.",
        
        # Test Case 2: Physics/Anatomy Question (Should trigger 5 concepts with many triples)
        "What is the main function of the human skeleton? A. to provide structural support and protect internal organs B. to produce energy C. to digest food D. to regulate body temperature",
        
        # Test Case 3: Broad scientific query triggering many abstract concepts and triples
        "Which phenomena are primarily responsible for the phases of the moon observed from Earth? A. Earth's rotation B. The moon's orbit around Earth C. the sun's rotation D. lunar eclipses"
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n\n{'='*80}")
        print(f"RUNNING TEST {i}/3")
        print(f"{'='*80}")
        
        # We need the base query (without MCQ) to simulate actual extraction pipeline
        base_query = q.split('?')[0] + '?'

        print("\n>>> Running SEQUENTIAL Filter...")
        t0 = time.time()
        res_seq = retriever_seq.search(query_text=base_query, top_k=5, original_query=q)
        t_seq = time.time() - t0

        print("\n>>> Running BATCH Filter...")
        t0 = time.time()
        res_batch = retriever_batch.search(query_text=base_query, top_k=5, original_query=q)
        t_batch = time.time() - t0

        print(f"\n>>> Time Comparison | Sequential: {t_seq:.2f}s | Batch: {t_batch:.2f}s | Speedup: {t_seq - t_batch:.2f}s")
        
        print_result_comparison(q, res_seq, res_batch)

    print("\nDone.")

if __name__ == "__main__":
    test_filter_accuracy()
