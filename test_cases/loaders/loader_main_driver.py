"""
Dataset Generation Pipeline Driver

Usage Examples:
1. Run all default benchmarks with default sample sizes (100 for most, 50 per MMLU subject):
   python loader_main_driver.py

2. Run specific benchmarks only:
   python loader_main_driver.py --benchmarks ai2_arc openbookqa

3. Run only one benchmark with a custom sample size:
   python loader_main_driver.py --benchmarks openbookqa --openbook-size 200

4. Override ALL standard sizes to a small number (great for quick testing):
   python loader_main_driver.py --all-size 10

Flags:
------
--benchmarks [list]       : List of benchmarks to run (e.g. ai2_arc mmlu openbookqa).
                            Default is all available benchmarks.
--all-size [int]          : Override ALL benchmark sizes to this value.
--ai2arc-size [int]       : Target approved sample size for AI2 ARC (default: 100).
--openbook-size [int]     : Target approved sample size for OpenBookQA (default: 100).
--mmlu-size [int]         : Target approved sample size for MMLU (default: 100).
"""
import argparse
import sys
import os

# Ensure we can import the loaders if running from root or inside test_cases/loaders
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from mmlu_loader import MMLULoader
from openbookqa_loader import OpenBookQALoader
from ai2_arc_loader import AI2_ARC_Loader

def main():
    parser = argparse.ArgumentParser(description="Generate approved datasets for GraphRAG pipelines.")
    
    parser.add_argument("--ai2arc-size", type=int, default=100, help="Target approved sample size for AI2 ARC (default: 100)")
    parser.add_argument("--mmlu-size", type=int, default=100, help="Target approved sample size for MMLU (default: 100)")
    parser.add_argument("--openbook-size", type=int, default=100, help="Target approved sample size for OpenBookQA (default: 100)")
    parser.add_argument("--all-size", type=int, help="Override all benchmark sizes with this value")
    parser.add_argument("--benchmarks", nargs='+', default=["ai2_arc", "mmlu", "openbookqa"], help="List of benchmarks to run (e.g. ai2_arc mmlu openbookqa)")
    
    args = parser.parse_args()
    
    # Normalize benchmarks to lowercase
    if args.benchmarks:
        args.benchmarks = [b.lower() for b in args.benchmarks]

    # Handle --all-size override
    if args.all_size:
        print(f"[Driver] Overriding sizes for all benchmarks to {args.all_size}...")
        args.ai2arc_size = args.all_size
        args.mmlu_size = args.all_size
        args.openbook_size = args.all_size

    print("="*60)
    print(f"Starting Dataset Generation Pipeline")
    print(f"Selected Benchmarks: {args.benchmarks}")
    print(f"AI2 ARC Target:     {args.ai2arc_size}")
    print(f"MMLU Target:        {args.mmlu_size}")
    print(f"OpenBookQA Target:  {args.openbook_size}")
    print("="*60)

    # 1. AI2 ARC
    if "ai2_arc" in args.benchmarks:
        print("\n>>> Running AI2 ARC Loader...")
        try:
            ai2arc_loader = AI2_ARC_Loader(sample_size=args.ai2arc_size)
            ai2arc_loader.generate_dataset()
        except Exception as e:
            print(f"[ERROR] AI2 ARC Loader failed: {e}")

    # 2. MMLU
    if "mmlu" in args.benchmarks:
        print("\n>>> Running MMLU Loader...")
        try:
            mmlu_loader = MMLULoader(sample_size=args.mmlu_size)
            mmlu_loader.generate_dataset()
        except Exception as e:
            print(f"[ERROR] MMLU Loader failed: {e}")

    # 3. OpenBookQA
    if "openbookqa" in args.benchmarks:
        print("\n>>> Running OpenBookQA Loader...")
        try:
            openbook_loader = OpenBookQALoader(sample_size=args.openbook_size)
            openbook_loader.generate_dataset()
        except Exception as e:
            print(f"[ERROR] OpenBookQA Loader failed: {e}")

    print("\n" + "="*60)
    print("All loaders finished.")
    print("="*60)

if __name__ == "__main__":
    main()
