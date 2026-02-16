"""
Hallucination Evaluation Pipeline - Main Driver

Orchestrates the full evaluation pipeline:
1. (Optional) Generate synthetic testset from documents
2. Run hallucination metrics (Faithfulness, Answer Relevance, Context Recall)
3. Run benchmark evaluators (HotpotQA, MMLU, TruthfulQA)
4. Run statistical analysis

Usage:
    python main_driver.py                    # Run full pipeline
    python main_driver.py --generate-testset # Generate testset first, then run
    python main_driver.py --metrics-only     # Run only hallucination metrics
"""
import os
import sys
import argparse

# Add project root and stats directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
pipeline_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if pipeline_root not in sys.path:
    sys.path.append(pipeline_root)
if current_dir not in sys.path:
    sys.path.append(current_dir)

import answer_relevance
import context_recall
import faithfulness

def run_testset_generation(test_size: int = 10):
    """Generate synthetic test cases using RAGAS."""
    print("=" * 60)
    print("STEP 0: Generating Synthetic Testset (RAGAS)")
    print("=" * 60)
    import generate_testset
    generate_testset.generate_synthetic_testset(test_size=test_size)

def run_hallucination_metrics():
    """Run all hallucination evaluation metrics."""
    print("\n" + "=" * 60)
    print("STEP 1: Running Hallucination Metrics")
    print("=" * 60)
    
    print("\n--- Faithfulness ---")
    try:
        faithfulness.main()
    except Exception as e:
        print(f"Faithfulness failed: {e}")
    
    print("\n--- Answer Relevance ---")
    try:
        answer_relevance.main()
    except Exception as e:
        print(f"Answer Relevance failed: {e}")
    
    print("\n--- Context Recall ---")
    try:
        context_recall.main()
    except Exception as e:
        print(f"Context Recall failed: {e}")

def run_benchmarks():
    """Run benchmark evaluators."""
    print("\n" + "=" * 60)
    print("STEP 2: Running Benchmark Evaluators")
    print("=" * 60)
    
    try:
        import hotpotqa
        print("\n--- HotpotQA ---")
        hotpotqa.main()
    except Exception as e:
        print(f"HotpotQA failed: {e}")
    
    try:
        import mmlu
        print("\n--- MMLU ---")
        mmlu.main()
    except Exception as e:
        print(f"MMLU failed: {e}")
    
    try:
        import truthfulqa
        print("\n--- TruthfulQA ---")
        truthfulqa.main()
    except Exception as e:
        print(f"TruthfulQA failed: {e}")

def run_statistical_analysis():
    """Run statistical tests on collected results."""
    print("\n" + "=" * 60)
    print("STEP 3: Running Statistical Analysis")
    print("=" * 60)
    
    try:
        from stats import stats_driver
        stats_driver.perform_statistical_tests()
    except Exception as e:
        print(f"Statistical analysis failed: {e}")

def run_hallucination_evaluation_pipeline(generate_testset: bool = False, 
                                           metrics_only: bool = False,
                                           test_size: int = 10):
    """
    Run the complete hallucination evaluation pipeline.
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_root, "hallucination_evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    if generate_testset:
        run_testset_generation(test_size=test_size)
    
    run_hallucination_metrics()
    
    if not metrics_only:
        run_benchmarks()
    
    run_statistical_analysis()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination Evaluation Pipeline")
    parser.add_argument("--generate-testset", action="store_true", 
                       help="Generate synthetic testset before evaluation")
    parser.add_argument("--metrics-only", action="store_true",
                       help="Run only hallucination metrics (skip benchmarks)")
    parser.add_argument("--test-size", type=int, default=10,
                       help="Number of synthetic test cases to generate")
    args = parser.parse_args()
    
    run_hallucination_evaluation_pipeline(
        generate_testset=args.generate_testset,
        metrics_only=args.metrics_only,
        test_size=args.test_size
    )