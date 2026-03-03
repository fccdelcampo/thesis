try:
    from .graphrag import graphrag_driver
    from .prolog import prolog_driver
    from .llm import generate, decide_fallback
    from .prompt_reconstructor import reconstruct_prompt
    from .graphrag.config import SKIP_LOGICAL_EVIDENCE_LLM
except ImportError:
    # Fallback for direct script execution
    from graphrag import graphrag_driver
    from prolog import prolog_driver
    from llm import generate
    from prompt_reconstructor import reconstruct_prompt
    from graphrag.config import SKIP_LOGICAL_EVIDENCE_LLM

def run_pipeline(user_prompt: str, flag: str, sample_mode: bool = False, include_conclusion: bool = True) -> dict:
    if flag != "q":
        pass
    
    # Separate inference run for deciding if pipeline would fall back to GraphRAG
    fallback = decide_fallback(user_prompt)

    graphrag_output = graphrag_driver.run_pipeline(user_question=user_prompt, fallback=fallback) if flag != "q" else None
    
    if flag != "q" and graphrag_output:
        # Print the intermediate GraphRAG logical evidence to the terminal for transparency
        print("\n" + "="*80)
        print("GRAPHRAG LOGICAL EVIDENCE")
        print("="*80)
        print(graphrag_output.get("answer", "No evidence found."))
        print("="*80 + "\n")
        print("DEBUG: Transferring evidence to Prolog engine...\n")
        query = graphrag_output.get("query", user_prompt)
        context_from_prompt = graphrag_output.get("text_context", "")
        retrieved_context = graphrag_output.get("contexts", [])  # raw RetrieverResultItem list
    else:
        query = user_prompt
        context_from_prompt = ""
        retrieved_context = []
        
    # ── Split KBPedia vs Wikidata retrieved items ────────────────────────────
    if isinstance(retrieved_context, list):
        kbpedia_items = []
        wikidata_items = []
        raw_context_strings = []
        for item in retrieved_context:
            src = ""
            if hasattr(item, "metadata"):
                src = (item.metadata or {}).get("source", "")
            elif isinstance(item, dict):
                src = item.get("metadata", {}).get("source", "")
            item_str = str(item)
            raw_context_strings.append(item_str)
            if src == "Wikidata":
                wikidata_items.append(item_str)
            else:
                kbpedia_items.append(item_str)

        if SKIP_LOGICAL_EVIDENCE_LLM:
            # Bypass the GRAPHRAG_TEMPLATE LLM step — pass raw triples directly to Prolog.
            # This is richer and avoids the intermediate LLM bottleneck.
            retrieved_context_str = "\n".join(raw_context_strings)
            print("DEBUG: SKIP_LOGICAL_EVIDENCE_LLM=True — passing raw triples to Prolog.")
        else:
            # Default: use the LLM-distilled logical evidence text as context
            retrieved_context_str = graphrag_output.get("answer", "") if graphrag_output else ""

        # Append Wikidata facts to context_from_prompt so they reach the synthesis LLM
        if wikidata_items:
            wikidata_block = "\n\nWikidata Background Facts (for context, not Prolog logic):\n" + "\n".join(wikidata_items)
            context_from_prompt = (context_from_prompt or "") + wikidata_block
    else:
        retrieved_context_str = str(retrieved_context) if retrieved_context else ""

    # graphrag_answer is always the LLM-generated answer (for the synthesis LLM fallback)
    graphrag_answer = graphrag_output.get("answer", "") if graphrag_output else ""
    pgr_results = {}
    if not fallback:
        pgr_results = prolog_driver.run_pipeline(user_question=user_prompt, retrieved_context=retrieved_context_str, context_from_prompt=context_from_prompt) if flag != "q" else None
    
    return {
        "answer": pgr_results['text_answer'] if pgr_results else "Error generating answer",
        "database": pgr_results.get("database", "") if pgr_results else "No database generated.",
        "query": pgr_results.get("query", "") if pgr_results else "No query generated.",
        "contexts": retrieved_context if isinstance(retrieved_context, list) else [retrieved_context] if retrieved_context else [],
        "context_from_prompt": context_from_prompt,
        "prolog_explanation": pgr_results.get("prolog_explanation", "") if pgr_results else "",
        "explainer_output": pgr_results.get("explainer_output", "") if pgr_results else "No explainer output generated.",
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