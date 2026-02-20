# Authenticate Hugging Face
from huggingface_hub import login
login(token="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")

import os
import json
import random
from datasets import load_dataset
try:
    from test_case_chooser import approve
except ImportError:
    # If run as standalone and test_case_chooser is sibling
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from test_case_chooser import approve

class FolioLoader:
    """
    Loader for the FOLIO dataset (yale-nlp/FOLIO).
    """
    
    # --- Configuration ---
    # Default path relative to project root or wherever it's run
    # If run from root, 'test_cases/folio.jsonl'
    FILE_PATH = 'test_cases/folio.jsonl'
    SAMPLE_SIZE = 50
    RANDOM_SEED = 42

    def __init__(self, sample_size=None):
        if sample_size is not None:
            self.SAMPLE_SIZE = sample_size
        
        # Ensure path is absolute to avoid confusion or relative path issues
        # Assume we want it in 'test_cases/folio.jsonl' relative to CWD if running driver from root
        self.file_path = self.FILE_PATH

    def ensure_data_exists(self):
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            return

        print(f"[FolioLoader] File {self.file_path} missing or empty. Generating now...")
        self._generate_dataset()

    def _generate_dataset(self):
        random.seed(self.RANDOM_SEED)
        all_selected_data = []

        print(f"[FolioLoader] Fetching 'yale-nlp/FOLIO' from Hugging Face...")
        
        try:
            # Load validation split
            ds = load_dataset("yale-nlp/FOLIO", split="validation", trust_remote_code=True)
            
            total_available = len(ds)
            print(f"[FolioLoader] Total available: {total_available}. Goal: {self.SAMPLE_SIZE} approved.")
            
            indices = list(range(total_available))
            random.shuffle(indices)
            
            approved_count = 0
            
            for idx in indices:
                if approved_count >= self.SAMPLE_SIZE:
                    break
                
                item = ds[idx]
                
                premises = item.get('premises', [])
                premises_text = " ".join(premises)
                conclusion = item.get('conclusion', '')
                label = item.get('label', 'UNKNOWN')
                
                # Format: "What can be concluded following the given premises? " + premises
                question_text = f"What can be concluded following the given premises? {premises_text}"
                
                entry = {
                    "question": question_text,
                    "answer": conclusion,
                    "label": label,
                    "premises": premises
                }
                
                user_prompt = f"Approving FOLIO:\nQ: {question_text}\nA: {conclusion} (Label: {label})"
                
                # Interactive approval
                if approve(user_prompt):
                    all_selected_data.append(entry)
                    approved_count += 1
                    print(f"   [+] Approved. Count: {approved_count}/{self.SAMPLE_SIZE}")
                    
        except Exception as e:
            print(f"[FolioLoader] FAILED to load: {e}")
            import traceback
            traceback.print_exc()

        if all_selected_data:
            os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                for entry in all_selected_data:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"[FolioLoader] Successfully saved {len(all_selected_data)} questions to {self.file_path}")
        else:
            print("[FolioLoader] No data selected/saved.")

    def generate_dataset(self):
        self._generate_dataset()

    def load_data(self):
        self.ensure_data_exists()
        data = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        data.append({
                            "dataset": "folio",
                            "question": item['question'],
                            "ground_truth": item['answer'],
                            "ground_truth_context": item.get('premises'), 
                            "label": item.get('label')
                        })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[FolioLoader] Error reading file: {e}")
            return []
        return data

if __name__ == "__main__":
    loader = FolioLoader()
    loader.generate_dataset()
