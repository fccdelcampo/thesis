try:
    from .graphrag import graphrag_driver
    from .prolog import prolog_driver
    from .llm import generate
    from .prompt_reconstructor import reconstruct_prompt
except ImportError:
    # Fallback for direct script execution
    from graphrag import graphrag_driver
    from prolog import prolog_driver
    from llm import generate
    from prompt_reconstructor import reconstruct_prompt

def run_pipeline(user_prompt: str, flag: str, sample_mode: bool = False, include_conclusion: bool = True) -> dict:
    if flag != "q":
        pass
        # try:
            # print("Auto-clearing stale data...")
            # graphrag_driver.clear_local_data()
        # except Exception as e:
        #     print(f"Warning: Failed to auto-clear data: {e}")

    graphrag_output = graphrag_driver.run_pipeline(user_prompt) if flag != "q" else None
    
    if flag != "q" and graphrag_output:
        # Print the intermediate GraphRAG logical evidence to the terminal for transparency
        print("\n" + "="*80)
        print("GRAPHRAG LOGICAL EVIDENCE")
        print("="*80)
        print(graphrag_output.get("answer", "No evidence found."))
        print("="*80 + "\n")
        print("DEBUG: Transferring evidence to Prolog engine...\n")
        # 'query' is not returned by graphrag_driver, so we use the user_prompt
        query = graphrag_output.get("query", user_prompt) 
        context_from_prompt = graphrag_output.get("text_context", "") 
        retrieved_context = graphrag_output.get("answer", [])
    else:
        query = user_prompt 
        context_from_prompt = ""
        retrieved_context = []
        
    # ── Split KBPedia vs Wikidata retrieved items ────────────────────────────
    # Wikidata plain-text triples (definition:, instance of:, subclass of:, ...)
    # are not structured Prolog-ready facts. Sending them to the Prolog generator
    # causes the LLM to awkwardly transliterate them into predicates, often poorly.
    # Instead: KBPedia items → Prolog generator; Wikidata items → synthesis LLM only.
    if isinstance(retrieved_context, list):
        kbpedia_items = []
        wikidata_items = []
        for item in retrieved_context:
            # RetrieverResultItem objects have a .metadata dict; plain strings have no source
            src = ""
            if hasattr(item, "metadata"):
                src = (item.metadata or {}).get("source", "")
            elif isinstance(item, dict):
                src = item.get("metadata", {}).get("source", "")
            if src == "Wikidata":
                wikidata_items.append(str(item))
            else:
                kbpedia_items.append(str(item))

        retrieved_context_str = "\n".join(kbpedia_items)

        # Append Wikidata facts to context_from_prompt so they reach the synthesis LLM
        if wikidata_items:
            wikidata_block = "\n\nWikidata Background Facts (for context, not Prolog logic):\n" + "\n".join(wikidata_items)
            context_from_prompt = (context_from_prompt or "") + wikidata_block
    else:
        retrieved_context_str = str(retrieved_context) if retrieved_context else ""

    graphrag_answer = graphrag_output.get("answer", "") if graphrag_output else ""
    pgr_results = prolog_driver.run_pipeline(user_question=user_prompt, retrieved_context=retrieved_context_str, context_from_prompt=context_from_prompt, graphrag_answer=graphrag_answer) if flag != "q" else None
 
    return {
        "answer": pgr_results['text_answer'] if pgr_results else "Error generating answer",
        "contexts": retrieved_context if isinstance(retrieved_context, list) else [retrieved_context] if retrieved_context else [],
        "context_from_prompt": context_from_prompt,
        "prolog_explanation": pgr_results.get("prolog_explanation", "") if pgr_results else "",
        "prolog_error": pgr_results.get("prolog_error") if pgr_results else "Prolog skipped"
    }
    
def run_graphrag_pipeline(user_prompt: str) -> dict:
    """Wrapper function to run the entire pipeline with a given prompt and flag."""
    return run_pipeline(user_prompt, flag="")

if __name__ == "__main__":
    prompt = "Which of the following parts of a plant cell has a function that is most similar to the function of an animal skeleton?"
    print(f"--- Running Pipeline with prompt: '{prompt}' ---")
    # flag default to empty string to run everything
    output = run_pipeline(prompt, flag="", include_conclusion=True)
    print("\n--- Final Answer ---")
    print(output.get("answer"))