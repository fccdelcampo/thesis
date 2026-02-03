from graphrag import graphrag_driver
from prolog import prolog_driver
from llm import generate
from prompt_reconstructor import reconstruct_prompt

def run_pipeline(user_question: str, include_conclusion: bool = True) -> dict:
    retrieved_context = graphrag_driver.run_pipeline()
    explainer_output = prolog_driver.run_pipeline(user_question, retrieved_context)
    final_prompt = reconstruct_prompt(user_question, retrieved_context, explanainer_output=explainer_output, include_conclusion=include_conclusion)
    return generate(final_prompt)