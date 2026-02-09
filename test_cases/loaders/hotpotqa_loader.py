import os
import json
import random
from datasets import load_dataset

class HotpotQALoader:
    # --- Configuration ---
    FILE_PATH = 'test-cases/hotpotqa.jsonl'
    SAMPLE_SIZE = 100
    RANDOM_SEED = 42
    SUBSET = "distractor" 

    def __init__(self):
        self.file_path = self.FILE_PATH

    def ensure_data_exists(self):
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            return

        print(f"[HotpotQALoader] File {self.file_path} missing or empty. Generating now...")
        self._generate_dataset()

    def _generate_dataset(self):
        random.seed(self.RANDOM_SEED)
        
        try:
            print(f"[HotpotQALoader] Fetching HotpotQA ({self.SUBSET}) from Hugging Face...")
            ds = load_dataset("hotpot_qa", self.SUBSET, split="validation", trust_remote_code=False)
            total_size = len(ds)
            print(f"[HotpotQALoader] Dataset loaded. Total size: {total_size}. Sampling {self.SAMPLE_SIZE}...")
            random_indices = random.sample(range(total_size), self.SAMPLE_SIZE)
            samples = ds.select(random_indices)
            
            formatted_data = []
            
            for item in samples:
                # HotpotQA Structure:
                # 'question': str
                # 'answer': str
                # 'context': {'title': [str, ...], 'sentences': [[str, ...], ...]} 
                # Note: In the HF 'hotpotqa' dataset, context is often a struct with parallel lists
                
                # Standardize Context immediately during generation
                # The HF dataset usually gives context as:
                # { 'title': ['Title A', 'Title B'], 'sentences': [['Sent A1', 'Sent A2'], ['Sent B1']] }
                
                raw_context = item['context']
                titles = raw_context['title']
                sentences_list = raw_context['sentences']
                
                flattened_context = []
                
                # Zip them together to create "Title: Sentence" strings
                for title, sentences in zip(titles, sentences_list):
                    for sent in sentences:
                        flattened_context.append(f"{title}: {sent}")

                entry = {
                    "dataset": "hotpotqa",
                    "question": item['question'],
                    "formatted_input": item['question'], # For Hotpot, input is just the question
                    "ground_truth": item['answer'],
                    "ground_truth_context": flattened_context,
                    "type": item['type'],
                    "level": item['level']
                }
                formatted_data.append(entry)

            # Save to disk
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                for entry in formatted_data:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"[HotpotQALoader] Successfully saved {len(formatted_data)} items to {self.file_path}")
                
        except Exception as e:
            print(f"[HotpotQALoader] CRITICAL ERROR: {e}")
            raise e

    def load_data(self):
        # 1. Auto-Generate if missing
        self.ensure_data_exists()
        
        # 2. Load
        data = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[HotpotQALoader] Error reading file: {e}")
            return []
            
        return data

if __name__ == "__main__":
    loader = HotpotQALoader()
    data = loader.load_data()
    print(f"Loaded {len(data)} items.")
    if data:
        print("Sample Question:", data[0]['question'])
        print("Sample Context Length:", len(data[0]['ground_truth_context']))