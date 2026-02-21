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

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Import our new interactive loaders
metrics_loaders_dir = os.path.join(project_root, "test_cases", "metrics_loaders")
if metrics_loaders_dir not in sys.path:
    sys.path.append(metrics_loaders_dir)

from faithfulness_loader import FaithfulnessLoader
from context_recall_loader import ContextRecallLoader
from answer_relevancy_loader import AnswerRelevancyLoader

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
    
    # 4. Generate - We generate 2x the requested size to allow room for user rejections
    generation_size = test_size * 2
    print(f"Generating {generation_size} synthetic test cases (Target approved: {test_size})...")
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=generation_size,
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
    
    # 6. Route to interactive loaders
    print("\nStarting Interactive Approval Process...")
    
    # Faithfulness
    f_loader = FaithfulnessLoader(sample_size=test_size)
    f_loader.generate_dataset(df)
    
    # Answer Relevancy
    ar_loader = AnswerRelevancyLoader(sample_size=test_size)
    ar_loader.generate_dataset(df)
    
    # Context Recall
    cr_loader = ContextRecallLoader(sample_size=test_size)
    cr_loader.generate_dataset(df)
    
    print("\nDone! Metrics datasets generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAGAS synthetic testset")
    parser.add_argument("--test_size", type=int, default=10, help="Target approved sample size for the metrics")
    args = parser.parse_args()
    
    generate_synthetic_testset(test_size=args.test_size)
