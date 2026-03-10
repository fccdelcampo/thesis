import argparse
import os
import glob

def clear_file(filepath, keep_header=False):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Read the first line if we need to keep the header
    header = ""
    if keep_header and filepath.endswith('.csv'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                header = f.readline()
        except Exception as e:
            print(f"Error reading header from {filepath}: {e}")
            return
            
    # Truncate the file and write the header back if applicable
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            if header:
                f.write(header)
                print(f"Cleared {filepath} (kept header)")
            else:
                print(f"Cleared {filepath}")
    except Exception as e:
        print(f"Error writing to {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Clear log and evaluation files while retaining CSV headers.")
    parser.add_argument("--pipeline", type=str, choices=["all", "prolog", "graphrag", "tuned"], default="all", help="Which pipeline logs to clear.")
    parser.add_argument("--benchmark", type=str, default="all", help="Which benchmark to clear (e.g., 'openbookqa', 'mmlu', 'all').")
    parser.add_argument("--metric", type=str, default="all", help="Which metric CSVs in hallucination_evaluation_results to clear (e.g., 'faithfulness', 'all').")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Clear JSONL logs in `logs/` directory
    pipeline_map = {
        "prolog": "prolog_graphrag_pipeline_logs",
        "graphrag": "graphrag_pipeline_logs",
        "tuned": "tuned_pipeline_logs"
    }
    
    pipelines_to_clear = pipeline_map.values() if args.pipeline == "all" else [pipeline_map[args.pipeline]]
    benchmarks_to_clear = "*" if args.benchmark == "all" else f"*{args.benchmark}*"
    
    for pipe in pipelines_to_clear:
        pattern = os.path.join(base_dir, "logs", pipe, f"{benchmarks_to_clear}.jsonl")
        for filepath in glob.glob(pattern):
            clear_file(filepath, keep_header=False)
            
    # 2. Clear test_cases/logs (approval logs)
    if args.pipeline == "all":
        approval_pattern = os.path.join(base_dir, "test_cases", "logs", f"{benchmarks_to_clear}.log")
        for filepath in glob.glob(approval_pattern):
            clear_file(filepath, keep_header=False)
            
    # 3. Clear metric scores (CSVs with headers) in `hallucination_evaluation_results/`
    if args.pipeline == "all": # Metric csvs usually combine all pipelines
        metrics_to_clear = "*" if args.metric == "all" else f"*{args.metric}*"
        # combine benchmark filter if specified, or if benchmark is 'all', ignore it for metric clearing except if metric is 'all'
        file_matcher = f"*{args.benchmark}*" if args.benchmark != "all" else f"{metrics_to_clear}*"
        if args.benchmark != "all" and args.metric != "all":
            file_matcher = f"*{args.benchmark}*{args.metric}*"
            
        csv_pattern = os.path.join(base_dir, "hallucination_evaluation_results", f"{file_matcher}.csv")
        for filepath in glob.glob(csv_pattern):
            clear_file(filepath, keep_header=True)
            
    # 4. Clear root log lines if they match
    if args.pipeline == "all" and args.benchmark == "all" and args.metric == "all":
        for txt in ["debug.txt", "debug_synthesis_llm.txt", "debug_rag_output.txt"]:
            fp = os.path.join(base_dir, txt)
            if os.path.exists(fp): clear_file(fp)

if __name__ == "__main__":
    main()
