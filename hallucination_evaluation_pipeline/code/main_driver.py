"""
Hallucination Evaluation Pipeline - Main Driver

Orchestrates the full evaluation pipeline:
1. (Optional) Generate synthetic testset from documents
2. Run hallucination metrics (Faithfulness, Answer Relevancy, Context Recall)
3. Run benchmark evaluators (AI2 ARC, OpenBookQA, MMLU)
4. Run statistical analysis

Usage:
    python hallucination_evaluation_pipeline\code\main_driver.py                    # Run full pipeline
    python hallucination_evaluation_pipeline\code\main_driver.py --generate-testset # Generate testset first, then run
    python hallucination_evaluation_pipeline\code\main_driver.py --metrics-only     # Run only hallucination metrics
    python hallucination_evaluation_pipeline\code\main_driver.py --benchmarks-only  # Run only benchmarks
    python hallucination_evaluation_pipeline\code\main_driver.py --test-size n      # Run with n test cases
    python hallucination_evaluation_pipeline\code\main_driver.py --benchmarks ai2_arc openbookqa mmlu      # Run specific benchmarks
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
    import test_cases.metrics_loaders.generate_testset as generate_testset
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
        answer_relevancy.main(test_size=test_size)
    except Exception as e:
        print(f"Answer Relevancy failed: {e}")
    
    print("\n--- Context Recall ---")
    try:
        context_recall.main(test_size=test_size)
    except Exception as e:
        print(f"Context Recall failed: {e}")

def run_benchmarks(test_size: int = 100, start_at: int = 0, benchmarks: list[str] = None):
    """Run benchmark evaluators with unified global start offset."""
    if benchmarks is None:
        benchmarks = ["ai2_arc", "openbookqa", "mmlu"]
        
    print("\n" + "=" * 60)
    print(f"STEP 2: Running Benchmark Evaluators: {benchmarks}")
    print("=" * 60)
    
    current_start_at = start_at
    
    if "ai2_arc" in benchmarks:
        if current_start_at >= test_size:
            print(f"\n--- AI2 ARC --- (Skipping entirely, start_at {current_start_at} >= test_size {test_size})")
            current_start_at -= test_size
        else:
            try:
                import ai2_arc
                print("\n--- AI2 ARC ---")
                ai2_arc.main(test_size=test_size, start_idx=current_start_at)
                current_start_at = 0
            except Exception as e:
                print(f"AI2 ARC failed: {e}")
                
    if "mmlu" in benchmarks:
        if current_start_at >= test_size:
            print(f"\n--- MMLU --- (Skipping entirely, start_at {current_start_at} >= test_size {test_size})")
            current_start_at -= test_size
        else:
            try:
                import mmlu
                print("\n--- MMLU ---")
                mmlu.main(test_size=test_size, start_idx=current_start_at)
                current_start_at = 0
            except Exception as e:
                print(f"MMLU failed: {e}")

    if "openbookqa" in benchmarks:
        if current_start_at >= test_size:
            print(f"\n--- OpenBookQA --- (Skipping entirely, start_at {current_start_at} >= test_size {test_size})")
            current_start_at -= test_size
        else:
            try:
                import openbookqa
                print("\n--- OpenBookQA ---")
                openbookqa.main(test_size=test_size, start_idx=current_start_at)
                current_start_at = 0
            except Exception as e:
                print(f"OpenBookQA failed: {e}")
        
    # The duplicate block of code below is removed as per the instruction's implied scope.

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
                                           start_at: int = 0,
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
        run_benchmarks(test_size=test_size, start_at=start_at, benchmarks=benchmarks)
    
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
                       help="Number of test cases to run per benchmark")
    parser.add_argument("--start-at", type=int, default=0,
                       help="Global offset to skip a number of test cases across the active benchmarks")
    parser.add_argument("--benchmarks", nargs='+', default=["ai2_arc", "openbookqa", "mmlu"],
                       help="List of benchmarks to run (e.g. ai2_arc openbookqa mmlu)")
    args = parser.parse_args()
    
    # Normalize benchmarks to lowercase
    benchmarks_lower = [b.lower() for b in args.benchmarks] if args.benchmarks else ["ai2_arc", "openbookqa", "mmlu"]
    
    run_hallucination_evaluation_pipeline(
        generate_testset=args.generate_testset,
        metrics_only=args.metrics_only,
        benchmarks_only=args.benchmarks_only,
        test_size=args.test_size,
        start_at=args.start_at,
        benchmarks=benchmarks_lower
    )