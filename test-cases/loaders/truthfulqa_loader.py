import os
import json
import random
from datasets import load_dataset

class TruthfulQALoader:
    FILE_PATH = 'test-cases/truthfulqa.jsonl'
    SAMPLE_SIZE = 100
    RANDOM_SEED = 42
    
    def __init__(self):
        self.file_path = self.FILE_PATH

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
            indices = random.sample(range(total), min(self.SAMPLE_SIZE, total))
            samples = ds.select(indices)
            
            formatted_data = []
            for item in samples:
                formatted_data.append({
                    "dataset": "truthfulqa",
                    "question": item['question'],
                    "formatted_input": item['question'],
                    "ground_truth": item['correct_answers'], 
                    "ground_truth_context": None
                })

            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                for entry in formatted_data:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"[TruthfulQALoader] Saved {len(formatted_data)} items with broad answer sets.")
                
        except Exception as e:
            print(f"[TruthfulQALoader] Error: {e}")
            raise e

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
    print(f"Loaded {len(data)} items.")
    print("Example Ground Truth (List):", data[0]['ground_truth'])