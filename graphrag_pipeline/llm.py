from neo4j_graphrag.generation.prompts import RagTemplate
from graphrag.config import GRAPHRAG_TEMPLATE
from neo4j_graphrag.generation import GraphRAG

def generate(llm, retriever, query) -> dict:
    rag_template = RagTemplate(template=GRAPHRAG_TEMPLATE, expected_inputs=['query_text', 'context'])

    graph_rag = GraphRAG(llm=llm, retriever=retriever, prompt_template=rag_template)

    graph_rag_results = graph_rag.search(
        query, 
        retriever_config={'top_k': 5}, 
        return_context=True
    )
    
    retrieved_context = graph_rag_results.retriever_result
    answer = graph_rag_results.answer

    output_file = "debug_rag_output.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(f" QUERY: {query}\n")
        f.write("="*50 + "\n\n")

        f.write(f"--- RETRIEVED CONTEXT (Top {len(retrieved_context.items)} Chunks) ---\n")
        
        for i, item in enumerate(retrieved_context.items):
            score = item.metadata.get('score', 0.0) if item.metadata else 0.0
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

    return {
        "answer": answer,
        "context": retrieved_context
    }