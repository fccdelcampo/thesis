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
        # 'query' is not returned by graphrag_driver, so we use the user_prompt
        query = graphrag_output.get("query", user_prompt) 
        context_from_prompt = graphrag_output.get("text_context", "") 
        retrieved_context = graphrag_output.get("answer", [])
    else:
        query = user_prompt 
        context_from_prompt = ""
        retrieved_context = []
        
    # Convert list to string for Prolog and LLM (they expect string input)
    retrieved_context_str = "\n".join(retrieved_context) if retrieved_context and isinstance(retrieved_context, list) else (str(retrieved_context) if retrieved_context is not None else "")

    print(f"DEBUG: retrieved_context_str type: {type(retrieved_context_str)}")
    print(f"DEBUG: query: {query}")

    explainer_output = prolog_driver.run_pipeline(user_question=query, retrieved_context=retrieved_context_str, context_from_prompt=context_from_prompt) if flag != "q" else None
 
    llm_result = generate(
        user_question=query, 
        retrieved_context=retrieved_context_str, 
        explainer_output=explainer_output["text_answer"] if explainer_output else "", 
        flag=flag,
        include_conclusion=include_conclusion,
        context_from_prompt=context_from_prompt
    )
    
    return {
        "answer": llm_result['text_answer'] if llm_result else "Error generating answer",
        "contexts": retrieved_context if isinstance(retrieved_context, list) else [retrieved_context] if retrieved_context else [],
        "context_from_prompt": context_from_prompt,
        "prolog_explanation": explainer_output.get("text_answer", "") if explainer_output else ""
    }
    
def run_graphrag_pipeline(user_prompt: str) -> dict:
    """Wrapper function to run the entire pipeline with a given prompt and flag."""
    retrieved_context = graphrag_driver.run_pipeline(user_prompt).get("answer", [])
    return retrieved_context

if __name__ == "__main__":
    prompt = "Which of the following parts of a plant cell has a function that is most similar to the function of an animal skeleton?"
    print(f"--- Running Pipeline with prompt: '{prompt}' ---")
    # flag default to empty string to run everything
    output = run_pipeline(prompt, flag="", include_conclusion=True)
    print("\n--- Final Answer ---")
    print(output.get("answer"))