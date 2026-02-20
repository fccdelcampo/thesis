"""
Dataset Generation Pipeline Driver

Usage Examples:
1. Run all default benchmarks with default sample sizes (100 for most, 50 per MMLU subject):
   python loader_main_driver.py

2. Run specific benchmarks only:
   python loader_main_driver.py --benchmarks hotpotqa truthfulqa

3. Run only one benchmark with a custom sample size:
   python loader_main_driver.py --benchmarks openbookqa --openbook-size 200

4. Override ALL standard sizes to a small number (great for quick testing):
   python loader_main_driver.py --all-size 10

Flags:
------
--benchmarks [list]       : List of benchmarks to run (e.g. hotpotqa truthfulqa mmlu folio openbookqa).
                            Default is all available benchmarks.
--all-size [int]          : Override ALL benchmark sizes to this value.
--hotpot-size [int]       : Target approved sample size for HotpotQA (default: 100).
--truthful-size [int]     : Target approved sample size for TruthfulQA (default: 100).
--folio-size [int]        : Target approved sample size for FOLIO (default: 100).
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

from hotpotqa_loader import HotpotQALoader
from mmlu_loader import MMLULoader
from truthfulqa_loader import TruthfulQALoader
from folio_loader import FolioLoader
from openbookqa_loader import OpenBookQALoader

def main():
    parser = argparse.ArgumentParser(description="Generate approved datasets for GraphRAG pipelines.")
    
    parser.add_argument("--hotpot-size", type=int, default=100, help="Target approved sample size for HotpotQA (default: 100)")
    parser.add_argument("--truthful-size", type=int, default=100, help="Target approved sample size for TruthfulQA (default: 100)")
    parser.add_argument("--mmlu-size", type=int, default=100, help="Target approved sample size for MMLU (default: 100)")
    parser.add_argument("--folio-size", type=int, default=100, help="Target approved sample size for FOLIO (default: 100)")
    parser.add_argument("--openbook-size", type=int, default=100, help="Target approved sample size for OpenBookQA (default: 100)")
    parser.add_argument("--all-size", type=int, help="Override all benchmark sizes with this value")
    parser.add_argument("--benchmarks", nargs='+', default=["mmlu", "folio", "openbookqa"], help="List of benchmarks to run (e.g. hotpotqa truthfulqa mmlu folio openbookqa)")
    
    args = parser.parse_args()
    
    # Normalize benchmarks to lowercase
    if args.benchmarks:
        args.benchmarks = [b.lower() for b in args.benchmarks]

    # Handle --all-size override
    if args.all_size:
        print(f"[Driver] Overriding sizes for all benchmarks to {args.all_size}...")
        args.hotpot_size = args.all_size
        args.truthful_size = args.all_size
        args.mmlu_size = args.all_size
        args.folio_size = args.all_size
        args.openbook_size = args.all_size

    print("="*60)
    print(f"Starting Dataset Generation Pipeline")
    print(f"Selected Benchmarks: {args.benchmarks}")
    print(f"HotpotQA Target:    {args.hotpot_size}")
    print(f"TruthfulQA Target:  {args.truthful_size}")
    print(f"MMLU Target:        {args.mmlu_size}")
    print(f"FOLIO Target:       {args.folio_size}")
    print(f"OpenBookQA Target:  {args.openbook_size}")
    print("="*60)

    # 1. HotpotQA
    if "hotpotqa" in args.benchmarks:
        print("\n>>> Running HotpotQA Loader...")
        try:
            hotpot_loader = HotpotQALoader(sample_size=args.hotpot_size)
            hotpot_loader.generate_dataset()
        except Exception as e:
            print(f"[ERROR] HotpotQA Loader failed: {e}")

    # 2. TruthfulQA
    if "truthfulqa" in args.benchmarks:
        print("\n>>> Running TruthfulQA Loader...")
        try:
            truthful_loader = TruthfulQALoader(sample_size=args.truthful_size)
            truthful_loader.generate_dataset()
        except Exception as e:
            print(f"[ERROR] TruthfulQA Loader failed: {e}")

    # 3. MMLU
    if "mmlu" in args.benchmarks:
        print("\n>>> Running MMLU Loader...")
        try:
            mmlu_loader = MMLULoader(sample_size=args.mmlu_size)
            mmlu_loader.generate_dataset()
        except Exception as e:
            print(f"[ERROR] MMLU Loader failed: {e}")

    # 4. FOLIO
    if "folio" in args.benchmarks:
        print("\n>>> Running FOLIO Loader...")
        try:
            folio_loader = FolioLoader(sample_size=args.folio_size)
            folio_loader.generate_dataset()
        except Exception as e:
            print(f"[ERROR] Folio Loader failed: {e}")

    # 5. OpenBookQA
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
