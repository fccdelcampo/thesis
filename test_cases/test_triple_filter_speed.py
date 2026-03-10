import time
import sys
import os
from pathlib import Path

# Add thesis root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import graphrag_pipeline.graphrag.graphrag_driver as gd
from prolog_graphrag_pipeline.graphrag.kbpedia_retriever import KBPediaRetriever

def test_triple_filter_speed():
    question = "Which of the following describes the fallacy of appeal to common practice? A. eliminating a source of evidence from consideration by claiming the source is flawed, when there is no true relationship between the alleged flaw and the reliability of the source B. reasoning that an act or series of acts should be carried out because they will have no significant consequences, when there is no evidence there will be no consequences C. drawing a conclusion after purposely leaving out evidence or ignoring known evidence that might contradict a claim D. arguing that something is the right thing to do based only on evidence that other people are already doing it."
    extracted_query = "Which of the following describes the fallacy of appeal to common practice?"
    
    print("Initializing LLM and Driver...")
    gd.ensure_driver_connected()
    encoder_llm, retriever_llm, embedder = gd.initialize_models()
    
    print("\n" + "="*50)
    print("TEST 1: ORIGINAL FAST MODE (hybrid_triple_filter=False)")
    print("="*50)
    retriever_fast = KBPediaRetriever(driver=gd.neo4j_driver, llm=retriever_llm, top_k=5)
    
    start_time = time.time()
    result_fast = retriever_fast.search(query_text=extracted_query, top_k=5, original_query=question)
    fast_duration = time.time() - start_time
    
    print(f"\n[FAST MODE] Time taken: {fast_duration:.2f} seconds")
    print(f"[FAST MODE] Items retrieved: {len(result_fast.items)}")
    
    print("\n" + "="*50)
    print("TEST 2: HYBRID MODE (hybrid_triple_filter=True)")
    print("="*50)
    retriever_rigorous = KBPediaRetriever(driver=gd.neo4j_driver, llm=retriever_llm, top_k=5)
    
    start_time = time.time()
    result_rigorous = retriever_rigorous.search(query_text=extracted_query, top_k=5, original_query=question)
    end_time = time.time()
    
    rigorous_duration = end_time - start_time
    print(f"\n[RIGOROUS MODE] Time taken: {rigorous_duration:.2f} seconds")
    print(f"[RIGOROUS MODE] Items retrieved: {len(result_rigorous.items)}")
    
    print("\n" + "="*50)
    print(f"CONCLUSION: Hybrid Mode was {rigorous_duration - fast_duration:.2f} seconds slower.")
    print("="*50)

if __name__ == "__main__":
    test_triple_filter_speed()
