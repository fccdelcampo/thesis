from datasets import load_dataset

def check_dataset(dataset_id, name):
    print(f"Checking {name} ({dataset_id})...")
    try:
        ds = load_dataset(dataset_id, split="validation", trust_remote_code=True)
        print(f"  [SUCCESS] Loaded {len(ds)} samples.")
        print("  Sample:", ds[0])
        return True
    except Exception as e:
        print(f"  [FAILED] {e}")
        return False

# Check FOLIO
check_dataset("yale-nlp/FOLIO", "FOLIO")

# Check ProofWriter (User suggested renma/ProofWriter)
if not check_dataset("renma/ProofWriter", "ProofWriter (renma)"):
    # Try alternatives if renma fails
    check_dataset("meta-eval/proofwriter", "ProofWriter (meta-eval)")
