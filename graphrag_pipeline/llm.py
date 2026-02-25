from neo4j_graphrag.generation.prompts import RagTemplate
try:
    from graphrag_pipeline.graphrag.config import GRAPHRAG_TEMPLATE
except ModuleNotFoundError:
    from graphrag.config import GRAPHRAG_TEMPLATE
from neo4j_graphrag.generation import GraphRAG

def generate(llm, retriever, query, original_query: str = "") -> dict:
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

    return {
        "answer": answer,
        "contexts": contexts
    }