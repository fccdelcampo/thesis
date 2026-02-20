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

def main():
    parser = argparse.ArgumentParser(description="Generate approved datasets for GraphRAG pipelines.")
    
    parser.add_argument("--hotpot-size", type=int, default=100, help="Target approved sample size for HotpotQA (default: 100)")
    parser.add_argument("--truthful-size", type=int, default=100, help="Target approved sample size for TruthfulQA (default: 100)")
    parser.add_argument("--mmlu-per-subject", type=int, default=50, help="Target approved samples per subject for MMLU (default: 50)")
    parser.add_argument("--folio-size", type=int, default=50, help="Target approved sample size for FOLIO (default: 50)")
    parser.add_argument("--all-size", type=int, help="Override Hotpot and Truthful sizes with this value (MMLU remains separate unless specified)")
    
    args = parser.parse_args()

    # Handle --all-size override
    if args.all_size:
        print(f"[Driver] Overriding HotpotQA and TruthfulQA sizes to {args.all_size}...")
        args.hotpot_size = args.all_size
        args.truthful_size = args.all_size
        args.folio_size = args.all_size

    print("="*60)
    print(f"Starting Dataset Generation Pipeline")
    print(f"HotpotQA Target:    {args.hotpot_size}")
    print(f"TruthfulQA Target:  {args.truthful_size}")
    print(f"MMLU Target (Subj): {args.mmlu_per_subject}")
    print(f"FOLIO Target:       {args.folio_size}")
    print("="*60)

    # 1. HotpotQA
    print("\n>>> Running HotpotQA Loader...")
    try:
        hotpot_loader = HotpotQALoader(sample_size=args.hotpot_size)
        hotpot_loader.generate_dataset()
    except Exception as e:
        print(f"[ERROR] HotpotQA Loader failed: {e}")

    # 2. TruthfulQA
    print("\n>>> Running TruthfulQA Loader...")
    try:
        truthful_loader = TruthfulQALoader(sample_size=args.truthful_size)
        truthful_loader.generate_dataset()
    except Exception as e:
        print(f"[ERROR] TruthfulQA Loader failed: {e}")

    # 3. MMLU
    print("\n>>> Running MMLU Loader...")
    try:
        mmlu_loader = MMLULoader(samples_per_subject=args.mmlu_per_subject)
        mmlu_loader.generate_dataset()
    except Exception as e:
        print(f"[ERROR] MMLU Loader failed: {e}")

    # 4. FOLIO
    print("\n>>> Running FOLIO Loader...")
    try:
        folio_loader = FolioLoader(sample_size=args.folio_size)
        folio_loader.generate_dataset()
    except Exception as e:
        print(f"[ERROR] Folio Loader failed: {e}")

    print("\n" + "="*60)
    print("All loaders finished.")
    print("="*60)

if __name__ == "__main__":
    main()
