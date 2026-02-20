import os
import json
import random
from datasets import load_dataset
from test_case_chooser import approve

class HotpotQALoader:
    # --- Configuration ---
    FILE_PATH = 'test_cases/hotpotqa.jsonl'
    SAMPLE_SIZE = 100
    RANDOM_SEED = 42
    SUBSET = "distractor" 

    def __init__(self, sample_size=None):
        self.file_path = self.FILE_PATH
        if sample_size is not None:
             self.SAMPLE_SIZE = sample_size

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
            print(f"[HotpotQALoader] Dataset loaded. Total size: {total_size}. Sampling until {self.SAMPLE_SIZE} approved items...")
            
            formatted_data = []
            seen_indices = set()
            
            # Loop until we have enough samples or exhausted the dataset
            while len(formatted_data) < self.SAMPLE_SIZE:
                # Stop if we've seen everyone
                if len(seen_indices) >= total_size:
                    print(f"[HotpotQALoader] Exhausted dataset! Only found {len(formatted_data)} approved items.")
                    break

                needed = self.SAMPLE_SIZE - len(formatted_data)
                # Heuristic: Fetch 3x what we need to account for rejections, but don't over-fetch too wildly
                fetch_size = min(needed * 3, total_size - len(seen_indices))
                # Clamp fetch_size to distinct reasonable chunks (e.g., at least 10 if possible)
                fetch_size = max(fetch_size, min(10, total_size - len(seen_indices)))
                
                # Generate unique indices
                # Note: For very large done sets, this random picking might get slower, but for 100/7000 it's fine.
                batch_indices = []
                while len(batch_indices) < fetch_size:
                     cand = random.randint(0, total_size - 1)
                     if cand not in seen_indices and cand not in batch_indices:
                         batch_indices.append(cand)
                
                seen_indices.update(batch_indices)
                
                print(f"[HotpotQALoader] Fetching batch of {len(batch_indices)}... (Progress: {len(formatted_data)}/{self.SAMPLE_SIZE})")
                
                # Load the actual data for this batch
                samples = ds.select(batch_indices)
                
                for idx_in_batch, item in enumerate(samples):
                    if len(formatted_data) >= self.SAMPLE_SIZE:
                        break

                    # HotpotQA Structure:
                    # 'question': str
                    # 'answer': str
                    # 'context': {'title': [str, ...], 'sentences': [[str, ...], ...]} 
                    
                    raw_context = item['context']
                    titles = raw_context['title']
                    sentences_list = raw_context['sentences']
                    
                    flattened_context = []
                    
                    for title, sentences in zip(titles, sentences_list):
                        for sent in sentences:
                            flattened_context.append(f"{title}: {sent}")

                    entry = {
                        "dataset": "hotpotqa",
                        "question": item['question'],
                        "formatted_input": item['question'], 
                        "ground_truth": item['answer'],
                        "ground_truth_context": flattened_context,
                        "type": item['type'],
                        "level": item['level']
                    }
                    
                    # Log approval process
                    # Note: 'items_left' is now relative to the target, not the batch
                    items_left = self.SAMPLE_SIZE - len(formatted_data)
                    currently_approved = len(formatted_data)
                    
                    with open("hotpotqa_approval.log", "a", encoding="utf-8") as log_file:
                        log_file.write(f"Approving: {item['question']} (Needed: {items_left}, Approved: {currently_approved})\n")
                        log_file.flush()
                        
                        if approve(item['question']):
                            formatted_data.append(entry)
                            log_file.write(f"Approved.\n")
                            log_file.flush()
                            print(f"   [+] Approved. Total: {len(formatted_data)}/{self.SAMPLE_SIZE}")

            # Save to disk
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                for entry in formatted_data:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"[HotpotQALoader] Successfully saved {len(formatted_data)} items to {self.file_path}")
                
        except Exception as e:
            print(f"[HotpotQALoader] CRITICAL ERROR: {e}")
            raise e
        
    def generate_dataset(self):
        self._generate_dataset()

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
    test_cases = loader.generate_dataset()