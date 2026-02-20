"""
Hallucination Evaluation Pipeline - Main Driver

Orchestrates the full evaluation pipeline:
1. (Optional) Generate synthetic testset from documents
2. Run hallucination metrics (Faithfulness, Answer Relevancy, Context Recall)
3. Run benchmark evaluators (HotpotQA, MMLU, TruthfulQA)
4. Run statistical analysis

Usage:
    python hallucination_evaluation_pipeline\code\main_driver.py                    # Run full pipeline
    python hallucination_evaluation_pipeline\code\main_driver.py --generate-testset # Generate testset first, then run
    python hallucination_evaluation_pipeline\code\main_driver.py --metrics-only     # Run only hallucination metrics
    python hallucination_evaluation_pipeline\code\main_driver.py --benchmarks-only  # Run only benchmarks
    python hallucination_evaluation_pipeline\code\main_driver.py --test-size n      # Run with n test cases
    python hallucination_evaluation_pipeline\code\main_driver.py --benchmarks mmlu folio openbookqa      # Run with n test cases
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

import answer_relevancy
import context_recall
import faithfulness

def run_testset_generation(test_size: int = 100):
    """Generate synthetic test cases using RAGAS."""
    print("=" * 60)
    print("STEP 0: Generating Synthetic Testset (RAGAS)")
    print("=" * 60)
    import generate_testset
    generate_testset.generate_synthetic_testset(test_size=test_size)


def run_hallucination_metrics(test_size: int = 100):
    """Run all hallucination evaluation metrics."""
    print("\n" + "=" * 60)
    print("STEP 1: Running Hallucination Metrics")
    print("=" * 60)
    
    print("\n--- Faithfulness ---")
    try:
        faithfulness.main(test_size=test_size)
    except Exception as e:
        print(f"Faithfulness failed: {e}")
    
    print("\n--- Answer Relevancy ---")
    try:
        answer_relevance.main(test_size=test_size)
    except Exception as e:
        print(f"Answer Relevancy failed: {e}")
    
    print("\n--- Context Recall ---")
    try:
        context_recall.main(test_size=test_size)
    except Exception as e:
        print(f"Context Recall failed: {e}")

def run_benchmarks(test_size: int = 100, benchmarks: list[str] = None):
    """Run benchmark evaluators."""
    if benchmarks is None:
        benchmarks = ["mmlu", "openbookqa", "folio"]
        
    print("\n" + "=" * 60)
    print(f"STEP 2: Running Benchmark Evaluators: {benchmarks}")
    print("=" * 60)
    
    if "hotpotqa" in benchmarks:
        try:
            import hotpotqa
            print("\n--- HotpotQA ---")
            hotpotqa.main(test_size=test_size)
        except Exception as e:
            print(f"HotpotQA failed: {e}")
    
    if "mmlu" in benchmarks:
        try:
            import mmlu
            print("\n--- MMLU ---")
            mmlu.main(test_size=test_size)
        except Exception as e:
            print(f"MMLU failed: {e}")
            
    if "truthfulqa" in benchmarks:
        try:
            import truthfulqa
            print("\n--- TruthfulQA ---")
            truthfulqa.main(test_size=test_size)
        except Exception as e:
            print(f"TruthfulQA failed: {e}")

    if "openbookqa" in benchmarks:
        try:
            import openbookqa
            print("\n--- OpenBookQA ---")
            openbookqa.main(test_size=test_size)
        except Exception as e:
            print(f"OpenBookQA failed: {e}")
            
    if "folio" in benchmarks:
        try:
            import folio
            print("\n--- FOLIO ---")
            folio.main(test_size=test_size)
        except Exception as e:
            print(f"FOLIO failed: {e}")

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
                                           benchmarks_only: bool = False,
                                           test_size: int = 100,
                                           benchmarks: list[str] = None):
    """
    Run the complete hallucination evaluation pipeline.
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_root, "hallucination_evaluation_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    if generate_testset:
        run_testset_generation(test_size=test_size)
    
    if not benchmarks_only:
        run_hallucination_metrics(test_size=test_size)
    
    if not metrics_only:
        run_benchmarks(test_size=test_size, benchmarks=benchmarks)
    
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
    parser.add_argument("--benchmarks-only", action="store_true",
                       help="Run only benchmarks (skip hallucination metrics)")
    parser.add_argument("--test-size", type=int, default=100,
                       help="Number of test cases to run/generate")
    parser.add_argument("--benchmarks", nargs='+', default=["mmlu", "openbookqa", "folio"],
                       help="List of benchmarks to run (e.g. mmlu openbookqa folio truthfulqa hotpotqa)")
    args = parser.parse_args()
    
    # Normalize benchmarks to lowercase
    benchmarks_lower = [b.lower() for b in args.benchmarks] if args.benchmarks else ["mmlu", "openbookqa", "folio"]
    
    run_hallucination_evaluation_pipeline(
        generate_testset=args.generate_testset,
        metrics_only=args.metrics_only,
        benchmarks_only=args.benchmarks_only,
        test_size=args.test_size,
        benchmarks=benchmarks_lower
    )