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
    graphrag_output = graphrag_driver.run_pipeline(user_prompt) if flag != "q" else None
    query = graphrag_output["query"] if flag != "q" else None
    context_from_prompt = graphrag_output["text_context"] if flag != "q" else None
    retrieved_context = graphrag_output["context"] if flag != "q" else None
    explainer_output = prolog_driver.run_pipeline(user_question=query, retrieved_context=retrieved_context, context_from_prompt=context_from_prompt) if flag != "q" else None
 
    llm_result = generate(
        user_question=query, 
        retrieved_context=retrieved_context, 
        explainer_output=explainer_output["text_answer"], 
        include_conclusion=include_conclusion
    )
    
    return {
        "answer": llm_result['text_answer'] if llm_result else "Error generating answer",
        "contexts": [retrieved_context] if retrieved_context else [],
        "context_from_prompt": context_from_prompt,
        "prolog_explanation": explainer_output.get("text_answer", "")
    }