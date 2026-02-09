from graphrag import graphrag_driver
from prolog import prolog_driver
from llm import generate
from prompt_reconstructor import reconstruct_prompt

def run_pipeline(user_prompt: str, include_conclusion: bool = True) -> dict:
    graphrag_output = graphrag_driver.run_pipeline(user_prompt)
    query = graphrag_output["query"]
    context_from_prompt = graphrag_output["text_context"]
    retrieved_context = graphrag_output["context"]
    explainer_output = prolog_driver.run_pipeline(user_question=query, retrieved_context=retrieved_context, context_from_prompt=context_from_prompt)
    final_prompt = reconstruct_prompt(user_question=query, retrieved_context=retrieved_context, explanainer_output=explainer_output["text_answer"], include_conclusion=include_conclusion)
    return generate(final_prompt)