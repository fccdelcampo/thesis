import os
import json
import random
from datasets import load_dataset

class MMLULoader:
    """
    A unified loader that auto-generates the MMLU dataset if it's missing,
    then loads and formats it for the GraphRAG pipeline.
    """
    
    # --- Configuration ---
    FILE_PATH = 'test-cases/mmlu.jsonl'
    SAMPLES_PER_SUBJECT = 10
    RANDOM_SEED = 42
    
    # The "Prolog-Educational" Subject List (Reasoning & Logic Focus)
    TARGET_SUBJECTS = [
        "formal_logic", 
        "abstract_algebra", 
        "college_computer_science", 
        "high_school_computer_science",
        "college_mathematics", 
        "high_school_mathematics",
        "college_physics", 
        "high_school_physics",
        "high_school_statistics", 
        "high_school_macroeconomics" 
    ]

    def __init__(self):
        self.file_path = self.FILE_PATH

    def ensure_data_exists(self):
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            return

        print(f"[MMLULoader] File {self.file_path} missing or empty. Generating now...")
        self._generate_dataset()

    def _generate_dataset(self):
        random.seed(self.RANDOM_SEED)
        all_selected_data = []

        print(f"[MMLULoader] Fetching {len(self.TARGET_SUBJECTS)} subjects from Hugging Face...")
        
        for subject in self.TARGET_SUBJECTS:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test")
                
                total_in_subject = len(ds)
                n_samples = min(total_in_subject, self.SAMPLES_PER_SUBJECT)
                
                if n_samples == 0:
                    continue

                random_indices = random.sample(range(total_in_subject), n_samples)
                
                samples = ds.select(random_indices)
                
                for item in samples:
                    # Map 0-3 to A-D
                    answer_key = ["A", "B", "C", "D"]
                    correct_char = answer_key[item['answer']]
                    
                    entry = {
                        "question": item['question'],
                        "options": item['choices'],
                        "answer": correct_char,
                        "subject": subject
                    }
                    all_selected_data.append(entry)
                    
            except Exception as e:
                print(f"[MMLULoader] FAILED to load {subject}: {e}")

        random.shuffle(all_selected_data)

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            for entry in all_selected_data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"[MMLULoader] Successfully saved {len(all_selected_data)} questions to {self.file_path}")

    def load_data(self):
        # 1. Auto-Generate if missing
        self.ensure_data_exists()
        
        # 2. Load and Format
        data = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        
                        # MMLU Logic: Format question + options
                        # From: ["Opt1", "Opt2"] -> "A) Opt1\nB) Opt2"
                        options = item.get('options', [])
                        formatted_options = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                        formatted_input = f"{item['question']}\n{formatted_options}"
                        
                        # Ground Truth is strictly the letter (e.g., 'A')
                        ground_truth = item.get('answer', '').strip()

                        data.append({
                            "dataset": "mmlu",
                            "question": item['question'],
                            "formatted_input": formatted_input,
                            "ground_truth": ground_truth,
                            "ground_truth_context": None, # Impossible for MMLU
                            "subject": item.get('subject', 'unknown')
                        })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[MMLULoader] Error reading file: {e}")
            return []
            
        return data

if __name__ == "__main__":
    loader = MMLULoader()
    data = loader.load_data()
    print(f"Loaded {len(data)} items.")
    if data:
        print("Sample Item:")
        print(json.dumps(data[0], indent=2))