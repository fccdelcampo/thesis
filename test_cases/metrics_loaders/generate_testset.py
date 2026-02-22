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
from langchain_ollama import ChatOllama, OllamaEmbeddings, ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Import our new interactive loaders
metrics_loaders_dir = os.path.join(project_root, "test_cases", "metrics_loaders")
if metrics_loaders_dir not in sys.path:
    sys.path.append(metrics_loaders_dir)

from metrics_loader import MetricsLoader

def load_documents():
    """Load all PDFs from the unprocessed_documents directory."""
    docs_dir = os.path.join(project_root, "documents")
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

from ragas.llms.prompt import Prompt
from ragas.testset.prompts import filter_question_prompt

def generate_synthetic_testset(test_size: int = 10):
    """Generate synthetic test cases using RAGAS TestsetGenerator."""
    
    # 1. Load Documents
    documents = load_documents()
    if not documents:
        print("No documents found. Cannot generate testset.")
        return
    
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
    from ragas.testset.graph import KnowledgeGraph
    from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, apply_transforms
    from ragas.testset import TestsetGenerator

    print("Initializing LLMs...")
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    critic_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # --- CUSTOM PROMPTS ADDED HERE ---
    
    # 1. Customize the Generator (Synthesizer)
    # We create a custom prompt for the multi-hop synthesizer to focus on logic chains
    logic_gen_prompt = Prompt(
        name="logic_chain_generator",
        instruction="""Generate a challenging question that requires both specific fact-retrieval 
        and multi-step logical reasoning. The answer must depend on a 'chain of impact' or 
        rule-based synthesis across multiple context nodes. Ensure the question is purely 
        logic-driven and does not explicitly ask for numerical calculations."""
    )
    
    multi_hop_synth = MultiHopSpecificQuerySynthesizer(llm=generator_llm)
    # Overriding the default prompt of the synthesizer
    multi_hop_synth.generate_query_prompt = logic_gen_prompt

    # 2. Customize the Critic (Filter)
    # We modify the global filter prompt used by the critic to enforce your specific constraints
    filter_question_prompt.instruction = """Assess if the question is solvable by an 
    inference engine using symbolic logic. REJECT any question that requires mathematical 
    calculations, solving for numerical variables, or building mathematical proofs. 
    Only approve questions that rely on the application of rules and facts."""

    # --- END OF CUSTOM PROMPTS ---

    distribution = [
        (multi_hop_synth, 0.8),
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.2)
    ]
    
    transforms = [
        HeadlinesExtractor(llm=generator_llm),
        HeadlineSplitter(max_tokens=1000)
    ]
    
    kg = KnowledgeGraph()
    apply_transforms(kg, transforms=transforms)
    
    print("Creating TestsetGenerator...")
    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        critic_llm=critic_llm,
        embedding_model=embeddings,
        knowledge_graph=kg
    )
    
    generation_size = test_size * 2
    print(f"Generating {generation_size} synthetic test cases (Target approved: {test_size})...")
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=generation_size,
        query_distribution=distribution
    )
    
    df = testset.to_pandas()
    
    output_csv = os.path.join(project_root, "test_cases", "synthetic_test_set.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved master dataset: {output_csv}")
    
    # 6. Route to interactive loaders
    print("\nStarting Interactive Approval Process...")
    
    # Metrics
    m_loader = MetricsLoader(sample_size=test_size)
    m_loader.generate_dataset(df)
    
    print("\nDone! Metrics datasets generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAGAS synthetic testset")
    parser.add_argument("--test_size", type=int, default=10, help="Target approved sample size for the metrics")
    args = parser.parse_args()
    
    generate_synthetic_testset(test_size=args.test_size)
