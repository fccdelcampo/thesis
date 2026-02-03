from neo4j_graphrag.generation.prompts import RagTemplate
from graphrag.config import GRAPHRAG_TEMPLATE

def generate(llm, retriever, query) -> dict: # -> str in Prolog-GraphRAG framework
    from neo4j_graphrag.generation.prompts import RagTemplate

    rag_template = RagTemplate(template=GRAPHRAG_TEMPLATE, expected_inputs=['query_text', 'context'])

    from neo4j_graphrag.generation import GraphRAG
    graph_rag = GraphRAG(llm=llm, retriever=retriever, prompt_template=rag_template)\

    graph_rag_results = graph_rag.search(query, retriever_config={'top_k': 5}, return_context=True)
    retrieved_context = graph_rag_results.retriever_result
    print(f"RESULTS:\n{retrieved_context}")
    answer = graph_rag_results.answer
    print(f"ANSWER:\n{answer}")
    return {
        "answer": answer,
        "context": retrieved_context
    } # Comment in Prolog-GraphRAG pipeline
    # return answer # Uncomment in Prolog-GraphRAG pipeline