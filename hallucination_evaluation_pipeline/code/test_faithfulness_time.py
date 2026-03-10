import time
import sys
import os
import statistics

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hallucination_evaluation_pipeline.code.fix_multiprocess_patch import apply_patch
apply_patch()

from hallucination_evaluation_pipeline.code.faithfulness import evaluate_with_ragas

def main():
    print("Starting RAGAS Faithfulness Benchmark (10 iterations)...")
    
    # Dummy data
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    contexts = ["Paris is the capital and most populous city of France."]
    
    times = []
    
    for i in range(10):
        print(f"\nIteration {i+1}/10:")
        start_time = time.time()
        
        try:
            score = evaluate_with_ragas(question, answer, contexts)
            end_time = time.time()
            duration = end_time - start_time
            times.append(duration)
            print(f"  Score: {score}")
            print(f"  External Time Measurement: {duration:.4f} seconds")
        except Exception as e:
            print(f"  Error: {e}")
            
    if times:
        avg_time = statistics.mean(times)
        print(f"\nBenchmark Complete.")
        print(f"Average Time: {avg_time:.4f} seconds")
        print(f"Min Time: {min(times):.4f} seconds")
        print(f"Max Time: {max(times):.4f} seconds")
    else:
        print("\nNo successful iterations to calculate average.")

if __name__ == "__main__":
    main()
