from neo4j_graphrag.generation.prompts import RagTemplate
try:
    from graphrag_pipeline.graphrag.config import GRAPHRAG_TEMPLATE
except ModuleNotFoundError:
    from graphrag.config import GRAPHRAG_TEMPLATE
from neo4j_graphrag.generation import GraphRAG

ENABLE_DUAL_INFERENCE_PASS = False

def generate(llm, retriever, query, original_query: str = "") -> dict:
    """
    Args:
        llm: The language model to use.
        retriever: The retriever to use.
        query (str): The question to be answered.
        original_query (str, optional): The original query. Defaults to "".

    Returns:
        dict: A dictionary of keys "answer" and "contexts", where "answer" is the final answer string and "contexts" is a list of retrieved context strings.
    """
    rag_template = RagTemplate(template=GRAPHRAG_TEMPLATE, expected_inputs=['query_text', 'context'])

    graph_rag = GraphRAG(llm=llm, retriever=retriever, prompt_template=rag_template)

    # Forward original_query (full MCQ prompt) so EnhancedRetriever's KBPedia filter
    # gets the complete question including answer options — same pattern as prolog_graphrag_pipeline.
    retriever_config = {'top_k': 5}
    if original_query:
        retriever_config['original_query'] = original_query

    graph_rag_results = graph_rag.search(
        query, 
        retriever_config=retriever_config, 
        return_context=True
    )
    
    first_run_answer = graph_rag_results.answer
    retrieved_context = graph_rag_results.retriever_result
    
    if ENABLE_DUAL_INFERENCE_PASS:
        # --- WIKIDATA INJECTION ---
        try:
            from graphrag_pipeline.graphrag.config import ENABLE_WIKIDATA_FETCHING
        except ModuleNotFoundError:
            from graphrag.config import ENABLE_WIKIDATA_FETCHING
            
        extracted_qids = set()
        if getattr(ENABLE_WIKIDATA_FETCHING, 'value', ENABLE_WIKIDATA_FETCHING):
            for item in retrieved_context.items:
                # Check metadata or content directly for Q-ID assigned by kbpedia_retriever
                if item.metadata and item.metadata.get('wikidata_qid'):
                    extracted_qids.add(item.metadata['wikidata_qid'])
                elif item.content:
                    import re
                    qid_match = re.search(r'\b(Q[1-9]\d*)\b', item.content)
                    if qid_match:
                        extracted_qids.add(qid_match.group(1))
                        
            if extracted_qids:
                try:
                    from graphrag_pipeline.graphrag.wikidata_retriever import WikidataRetriever
                except ModuleNotFoundError:
                    from graphrag.wikidata_retriever import WikidataRetriever
                    
                # print(f"DEBUG GRAPHRAG:Found {len(extracted_qids)} Wikidata Q-IDs. Fetching hybrid facts...")
                wd_fetcher = WikidataRetriever()
                
                # Since generate() is synchronous, run an enclosed async loop for the fetcher
                import asyncio
                async def fetch_all_facts():
                    tasks = [wd_fetcher.get_wikidata_facts(qid) for qid in extracted_qids]
                    return await asyncio.gather(*tasks, return_exceptions=True)
                    
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                if loop.is_running():
                    # We are already in an event loop (e.g. jupyter), use a synchronous run trick
                    import nest_asyncio
                    nest_asyncio.apply()
                    wd_results = loop.run_until_complete(fetch_all_facts())
                else:
                    wd_results = asyncio.run(fetch_all_facts())
                
                # Create a synthetic result item to hold the Wikidata facts at the very top of context
                combined_wd_text = ""
                for res in wd_results:
                    if isinstance(res, str) and res.strip():
                        combined_wd_text += res + "\n"
                        
                if combined_wd_text:
                    from neo4j_graphrag.types import RetrieverResultItem
                    wd_item = RetrieverResultItem(
                        content=f"WIKIDATA FACTS:\n{combined_wd_text.strip()}",
                        metadata={"source": "wikidata_hybrid"}
                    )
                    retrieved_context.items.insert(0, wd_item)
        # --- END WIKIDATA INJECTION ---
        
        # We need to re-run graph_rag generation because we mutated the context but graph_rag already fired the LLM.
        # To fix this cleanly without rewriting neo4j_graphrag, we just build the string manually and call the underlying LLM.
        # The neo4j_graphrag implementation is basic so we can bypass it easily.
        contexts_strings = [item.content for item in retrieved_context.items if item.content]
        context_blob = "\n\n".join(contexts_strings)
        
        if ENABLE_DUAL_INFERENCE_PASS:
            prompt = GRAPHRAG_TEMPLATE.format(query_text=query, context=context_blob)
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

        output_file = "debug_rag_output.txt"
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(f" QUERY: {query}\n")
            f.write("="*50 + "\n\n")

            f.write(f"--- RETRIEVED CONTEXT (Top {len(retrieved_context.items)} Chunks) ---\n")
            
            for i, item in enumerate(retrieved_context.items):
                score = item.metadata.get('score', 0.0) if item.metadata else 0.0
                
                # Add Metadata Dump
                import json
                f.write(f"\n[Metadata Dump]\n{json.dumps(item.metadata, indent=2)}\n")
                
                # Clean newlines for readability in the text file
                clean_content = item.content.replace('\n', ' ')
                
                f.write(f"\n[Chunk {i+1} | Score: {score:.4f}]\n")
                f.write(f"{clean_content}\n")
                f.write("-" * 50 + "\n")

            f.write("\n" + "="*50 + "\n")
            f.write(" LLM ANSWER:\n")
            f.write("="*50 + "\n")
            f.write(f"{answer}\n")
            f.write("="*50 + "\n")

        print(f"Done. Check '{output_file}' for the context and answer.")

        # Extract text content from RetrieverResult items to ensure JSON serializability
        contexts = [item.content for item in retrieved_context.items]

    final_answer = first_run_answer if not ENABLE_DUAL_INFERENCE_PASS else answer
    final_contexts = retrieved_context if not ENABLE_DUAL_INFERENCE_PASS else contexts
    
    return {
        "answer": final_answer,
        "contexts": final_contexts
    }