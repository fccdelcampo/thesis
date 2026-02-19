"""
RAGAS Synthetic Testset Generator

Generates synthetic evaluation datasets from the local document corpus
using RAGAS TestsetGenerator. Outputs per-metric JSONL files to test_cases/.

Usage:
    python generate_testset.py [--test_size N]
"""
import os
import sys
import json
import argparse
from test_case_chooser import approve

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings

def load_documents():
    """Load all PDFs from the unprocessed_documents directory."""
    docs_dir = os.path.join(project_root, "unprocessed_documents")
    all_docs = []
    
    if not os.path.exists(docs_dir):
        print(f"ERROR: Documents directory not found: {docs_dir}")
        return all_docs
    
    for filename in os.listdir(docs_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_dir, filename)
            print(f"Loading: {filename}")
            try:
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  -> {len(docs)} pages loaded.")
            except Exception as e:
                print(f"  -> Failed to load {filename}: {e}")
    
    print(f"Total documents loaded: {len(all_docs)}")
    return all_docs

def generate_synthetic_testset(test_size: int = 10):
    """Generate synthetic test cases using RAGAS TestsetGenerator."""
    
    # 1. Load Documents
    documents = load_documents()
    if not documents:
        print("No documents found. Cannot generate testset.")
        return
    
    # 2. Initialize LLMs and Embeddings with RAGAS wrappers
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    print("Initializing LLMs...")
    generator_llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b-instruct-q4_K_M"))
    critic_llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b-instruct-q4_K_M"))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    
    # 3. Create TestsetGenerator
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context
    
    print("Creating TestsetGenerator...")
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )
    
    # 4. Generate
    print(f"Generating {test_size} test cases...")
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=test_size,
        distributions={
            simple: 0.5,
            reasoning: 0.25,
            multi_context: 0.25
        }
    )
    
    # 5. Convert to DataFrame and save
    df = testset.to_pandas()
    
    # Save master CSV
    output_csv = os.path.join(project_root, "test_cases", "synthetic_test_set.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved master dataset: {output_csv}")
    
    # 6. Generate per-metric JSONL files
    test_cases_dir = os.path.join(project_root, "test_cases")
    
    # Faithfulness: needs question (answer & context come from pipeline at eval time)
    faithfulness_path = os.path.join(test_cases_dir, "faithfulness.jsonl")
    with open(faithfulness_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            question = row.get("question, ")
            if approve(question):
                entry = {"question": question}
                f.write(json.dumps(entry) + "\n")
    print(f"Saved: {faithfulness_path} ({len(df)} entries)")
    
    # Answer Relevancy: needs question
    ar_path = os.path.join(test_cases_dir, "answer_relevancy.jsonl")
    with open(ar_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            entry = {"question": row.get("question", "")}
            f.write(json.dumps(entry) + "\n")
    print(f"Saved: {ar_path} ({len(df)} entries)")
    
    # Context Recall: needs question + ground_truth
    cr_path = os.path.join(test_cases_dir, "context_recall.jsonl")
    with open(cr_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            entry = {
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth", "")
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Saved: {cr_path} ({len(df)} entries)")
    
    print("\nDone! Testset generation complete.")
    print(f"Generated {len(df)} test cases across 3 metric files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAGAS synthetic testset")
    parser.add_argument("--test_size", type=int, default=10, help="Number of test cases to generate")
    args = parser.parse_args()
    
    generate_synthetic_testset(test_size=args.test_size)
