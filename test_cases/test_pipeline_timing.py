import time
import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prolog_graphrag_pipeline.graphrag.encoder import extract_query_and_context
from prolog_graphrag_pipeline.graphrag.graphrag_driver import initialize_models, ensure_driver_connected
import prolog_graphrag_pipeline.graphrag.graphrag_driver as graphrag_driver
from prolog_graphrag_pipeline.prolog.prolog_generator import generate_prolog_code
from prolog_graphrag_pipeline.prolog.explainer import generate_safe_scasp_wrapper, generate_explanation
from prolog_graphrag_pipeline.prolog.prolog_driver import use_scasp
from prolog_graphrag_pipeline import llm
import janus_swi as janus
from neo4j_graphrag.types import RetrieverResult
from prolog_graphrag_pipeline.prolog.prolog_llms import warmup_prolog_model

def run_timing_test(iteration, total_times):
    encoder_llm, retriever_llm, embedder = initialize_models()
    ensure_driver_connected()
    
    # Needs to be imported after setup due to dependencies
    from prolog_graphrag_pipeline.graphrag.kbpedia_retriever import KBPediaRetriever
    
    question = "The Treaty of Tordesillas divided newly discovered lands outside Europe between which two empires? A. Spanish Empire and Portuguese Empire B. British Empire and French Empire C. Ottoman Empire and Russian Empire D. Dutch Empire and Spanish Empire"
    
    print(f"\n{'='*80}")
    print(f"STARTING TIMING TEST (Iteration {iteration}: {'COLD CACHE (Warmup needed)' if iteration == 1 else 'WARM CACHE (Fast)'})")
    print(f"{'='*80}")
    
    times = {}
    
    # Optional: Warmup Prolog KV Cache before the actual request starts (just like the real pipeline)
    print("0. Priming KV Cache (if enabled)...")
    t_w0 = time.perf_counter()
    warmup_prolog_model()
    t_w1 = time.perf_counter()
    w_time = t_w1 - t_w0
    if iteration == 1:
        times['0. Prolog KV Cache Warmup'] = w_time
        print(f"[Done] 0. Prolog KV Cache Warmup: {w_time:.2f}s")
    
    # 1. Query & Context Extraction
    t0 = time.perf_counter()
    query, text_context = extract_query_and_context(question)
    t1 = time.perf_counter()
    times['1. Extraction (LLM)'] = t1 - t0
    print(f"[Done] 1. Extraction: {times['1. Extraction (LLM)']:.2f}s")
    
    # 2. KBPedia Retrieval (Entity -> DB -> LLM Batch Filter)
    t2 = time.perf_counter()
    kbpedia_retriever = KBPediaRetriever(driver=graphrag_driver.neo4j_driver, llm=retriever_llm, top_k=5)
    
    # Break down KBPedia into DB vs Filter
    t2_1 = time.perf_counter()
    import re
    entities = kbpedia_retriever.extract_entities(query)
    seen_uris = set()
    candidates = []
    
    for concept_name in entities:
        matches = kbpedia_retriever.find_concepts(concept_name, limit=2)
        if not matches: continue
        for match in matches:
            uri = match["uri"]
            if uri in seen_uris: continue
            seen_uris.add(uri)
            
            raw_triples = []
            if match["definition"]: raw_triples.append(f"definition: {match['definition'][:200]}")
            raw_triples.extend(kbpedia_retriever.get_neighborhood(uri))
            mcq_pattern = re.compile(r'^[A-D][.):] ', re.IGNORECASE)
            raw_triples = [t for t in raw_triples if not mcq_pattern.match(t)]
            
            if not raw_triples: continue
            candidates.append({
                "name": match["name"], "uri": uri,
                "definition": match.get("definition", ""), "triples": raw_triples,
            })
            
    t2_2 = time.perf_counter()
    times['2a. Entity Ext. & DB Lookup'] = t2_2 - t2_1
    print(f"[Done] 2a. Entity Ext. & DB Lookup: {times['2a. Entity Ext. & DB Lookup']:.2f}s")
    
    if candidates:
        filtered_candidates = kbpedia_retriever._filter_concepts_once(query, candidates, original_query=question)
        filtered_concepts_to_use = filtered_candidates[:5]
        t2_3 = time.perf_counter()
        times['2b. Top-K Selection (LLM)'] = t2_3 - t2_2
        print(f"[Done] 2b. Top-K Selection (LLM): {times['2b. Top-K Selection (LLM)']:.2f}s")
        
        if filtered_concepts_to_use:
            batch_data = [{"name": c["name"], "triples": c["triples"]} for c in filtered_concepts_to_use]
            batch_results = kbpedia_retriever.filter_triples_batch(query, batch_data, original_query=question)
            for c in filtered_concepts_to_use:
                selected = batch_results.get(c["name"], c["triples"])
                c["triples"] = selected
        t2_4 = time.perf_counter()
        times['2c. Triple Filter (LLM)'] = t2_4 - t2_3
        print(f"[Done] 2c. Triple Filter (LLM): {times['2c. Triple Filter (LLM)']:.2f}s")
    else:
        times['2b. Top-K Selection (LLM)'] = 0.0
        times['2c. Triple Filter (LLM)'] = 0.0

    # Execute actual search for string building to simulate real run
    kb_result = kbpedia_retriever.search(query, top_k=5, original_query=question)
    retrieved_context_str = "\n".join([item.content for item in kb_result.items]) if kb_result and kb_result.items else ""
    
    # 3. Prolog Generation
    t4 = time.perf_counter()
    database, prolog_query = generate_prolog_code(user_question=question, retrieved_context=retrieved_context_str, most_recent_error=None)
    t5 = time.perf_counter()
    times['3. Prolog DB Generation (LLM)'] = t5 - t4
    print(f"[Done] 3. Prolog DB Generation (LLM): {times['3. Prolog DB Generation (LLM)']:.2f}s")
    
    # 4. s(CASP) Inference
    t6 = time.perf_counter()
    use_scasp()
    wrapper = generate_safe_scasp_wrapper(prolog_query)
    janus.consult("user", database + "\n" + wrapper)
    result = janus.query_once("explain(Explanation).")
    human_readable_explanation = result.get('Explanation', '') if result else ''
    retrieved_values = str(result)
    t7 = time.perf_counter()
    times['4. s(CASP) Logic Inference'] = t7 - t6
    print(f"[Done] 4. s(CASP) Logic Inference: {times['4. s(CASP) Logic Inference']:.2f}s")
    
    # 5. Explainer LLM
    t8 = time.perf_counter()
    explainer_output = generate_explanation(
        user_question=question,
        retrieved_context=retrieved_context_str,
        database=database,
        query=prolog_query,
        retrieved_values=retrieved_values,
        human_readable_explanation=human_readable_explanation
    )
    t9 = time.perf_counter()
    times['5. s(CASP) Tree Explainer (LLM)'] = t9 - t8
    print(f"[Done] 5. s(CASP) Tree Explainer (LLM): {times['5. s(CASP) Tree Explainer (LLM)']:.2f}s")
    
    # 6. Final Synthesis LLM
    t10 = time.perf_counter()
    final_answer = llm.generate(
        user_question=question,
        retrieved_context=retrieved_context_str,
        explainer_output=explainer_output,
        flag="synthesis"
    )
    t11 = time.perf_counter()
    times['6. Final Synthesis (LLM)'] = t11 - t10
    print(f"[Done] 6. Final Synthesis (LLM): {times['6. Final Synthesis (LLM)']:.2f}s")
    
    total_time = sum(times.values())
    
    print(f"\n{'='*80}")
    print(f"        PIPELINE TIMING BREAKDOWN (Iteration {iteration})")
    print(f"{'='*80}")
    for k, v in times.items():
        percentage = (v / total_time) * 100
        print(f"{k.ljust(35)}: {v:>7.2f}s ({percentage:>5.1f}%)")
        
        # Accumulate
        if k not in total_times:
            total_times[k] = []
        total_times[k].append(v)
        
    print("-" * 80)
    print(f"{'Total Measured Time'.ljust(35)}: {total_time:>7.2f}s\n")

if __name__ == "__main__":
    total_times = {}
    
    # Run twice: once to trigger the cold cache computation, next to show the actual speed
    for i in range(1, 4):
        run_timing_test(i, total_times)
        
    print(f"\n{'='*80}")
    print(f"        AVERAGE LATENCY ACROSS RUNS")
    print(f"{'='*80}")
    for k, v in total_times.items():
        avg = sum(v) / len(v)
        print(f"{k.ljust(35)}: Avg {avg:>7.2f}s")
