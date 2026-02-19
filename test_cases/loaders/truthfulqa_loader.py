import os
import json
import random
from datasets import load_dataset
from test_case_chooser import approve

class TruthfulQALoader:
    FILE_PATH = 'test_cases/truthfulqa.jsonl'
    SAMPLE_SIZE = 100
    RANDOM_SEED = 42
    
    def __init__(self, sample_size=None):
        self.file_path = self.FILE_PATH
        if sample_size is not None:
             self.SAMPLE_SIZE = sample_size

    def ensure_data_exists(self):
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            return
        print(f"[TruthfulQALoader] Generating {self.FILE_PATH}...")
        self._generate_dataset()

    def _generate_dataset(self):
        random.seed(self.RANDOM_SEED)
        try:
            ds = load_dataset("truthful_qa", "generation", split="validation")
            
            total = len(ds)
            print(f"[TruthfulQALoader] Dataset loaded. Total: {total}. Goal: {self.SAMPLE_SIZE} approved.")
            
            formatted_data = []
            seen_indices = set()
            
            while len(formatted_data) < self.SAMPLE_SIZE:
                 if len(seen_indices) >= total:
                     print(f"[TruthfulQALoader] Exhausted dataset! Found {len(formatted_data)}.")
                     break
                 
                 needed = self.SAMPLE_SIZE - len(formatted_data)
                 fetch_size = min(needed * 3, total - len(seen_indices))
                 fetch_size = max(fetch_size, min(10, total - len(seen_indices)))
                 
                 batch_indices = []
                 while len(batch_indices) < fetch_size:
                     cand = random.randint(0, total - 1)
                     if cand not in seen_indices and cand not in batch_indices:
                         batch_indices.append(cand)
                 
                 seen_indices.update(batch_indices)
                 
                 samples = ds.select(batch_indices)
                 
                 for idx_in_batch, item in enumerate(samples):
                        if len(formatted_data) >= self.SAMPLE_SIZE:
                            break
                        
                        # Start logging to file
                        with open("truthfulqa_approval.log", "a", encoding="utf-8") as log_file:
                            items_left = self.SAMPLE_SIZE - len(formatted_data)
                            currently_approved = len(formatted_data)
                            
                            log_file.write(f"Approving: {item['question']} (Needed: {items_left}, Approved: {currently_approved})\n")
                            log_file.flush()

                            if approve(item['question']):
                                log_file.write(f"Approved: {item['question']}\n")
                                log_file.flush()
                                formatted_data.append({
                                    "dataset": "truthfulqa",
                                    "question": item['question'],
                                    "formatted_input": item['question'],
                                    "ground_truth": item['correct_answers'], 
                                    "ground_truth_context": None
                                })
                                print(f"   [+] Approved. Total: {len(formatted_data)}/{self.SAMPLE_SIZE}")

            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                for entry in formatted_data:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"[TruthfulQALoader] Saved {len(formatted_data)} items with broad answer sets.")
                
        except Exception as e:
            print(f"[TruthfulQALoader] Error: {e}")
            raise e
        
    def generate_dataset(self):
        self._generate_dataset()

    def load_data(self):
        self.ensure_data_exists()
        data = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[TruthfulQALoader] Read Error: {e}")
            return []
        return data

if __name__ == "__main__":
    loader = TruthfulQALoader()
    data = loader.load_data()
    test_cases = loader.generate_dataset()