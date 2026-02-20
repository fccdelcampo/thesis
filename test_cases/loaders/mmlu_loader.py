import os
import json
import random
from datasets import load_dataset
from test_case_chooser import approve

class MMLULoader:
    """
    A unified loader that auto-generates the MMLU dataset if it's missing,
    then loads and formats it for the GraphRAG pipeline.
    """
    
    # --- Configuration ---
    FILE_PATH = 'test_cases/mmlu.jsonl'
    SAMPLES_PER_SUBJECT = 50
    RANDOM_SEED = 42
    
    # The "Prolog-Educational" Subject List (Reasoning & Logic Focus)
    TARGET_SUBJECTS = [
        "machine_learning",
        "computer_security",
        "college_computer_science",
        "formal_logic",
        "high_school_microeconomics",
        "nutrition",
        "professional_medicine",
        "conceptual_physics",
        "high_school_chemistry",
        "logical_fallacies"    
    ]

    def __init__(self, samples_per_subject=None):
        self.file_path = self.FILE_PATH
        if samples_per_subject is not None:
            self.SAMPLES_PER_SUBJECT = samples_per_subject

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
                total_in_subject = len(ds)
                print(f"[MMLULoader] Subject '{subject}': Total available {total_in_subject}. Goal: {self.SAMPLES_PER_SUBJECT} approved.")
                
                subject_approved_count = 0
                seen_indices = set()
                
                while subject_approved_count < self.SAMPLES_PER_SUBJECT:
                    # Stop if we've seen everyone
                    if len(seen_indices) >= total_in_subject:
                        print(f"[MMLULoader] Exhausted subject '{subject}'! Found {subject_approved_count} approved.")
                        break

                    needed = self.SAMPLES_PER_SUBJECT - subject_approved_count
                    fetch_size = min(needed * 3, total_in_subject - len(seen_indices))
                    fetch_size = max(fetch_size, min(10, total_in_subject - len(seen_indices)))

                    batch_indices = []
                    while len(batch_indices) < fetch_size:
                        cand = random.randint(0, total_in_subject - 1)
                        if cand not in seen_indices and cand not in batch_indices:
                            batch_indices.append(cand)
                    
                    seen_indices.update(batch_indices)
                    
                    samples = ds.select(batch_indices)
                    
                    for idx_in_batch, item in enumerate(samples):
                        if subject_approved_count >= self.SAMPLES_PER_SUBJECT:
                            break

                        # Map 0-3 to A-D
                        answer_key = ["A", "B", "C", "D"]
                        correct_char = answer_key[item['answer']]
                        
                        entry = {
                            "question": item['question'],
                            "options": item['choices'],
                            "answer": correct_char,
                            "subject": subject
                        }
                        # Start logging to file
                        with open("mmlu_approval.log", "a", encoding="utf-8") as log_file:
                            items_left = self.SAMPLES_PER_SUBJECT - subject_approved_count
                            # curently_approved across ALL subjects? Or just this one? 
                            # The user prompt wanted "hits the 100 approved wanted size" (implied per benchmark or total?)
                            # For MMLU it is per subject. 
                            # Let's log totals.
                            total_approved_so_far = len(all_selected_data) + subject_approved_count
                            
                            log_file.write(f"Approving ({subject}): {item['question']} (Needed: {items_left}, Total Approved: {total_approved_so_far})\n")
                            log_file.flush()
                            
                            if approve(item['question'] + "? Choices: " + " ".join(item['choices'])):
                                all_selected_data.append(entry)
                                subject_approved_count += 1
                                log_file.write(f"\n")
                                log_file.flush()
                                print(f"   [+] Approved ({subject}). Subject count: {subject_approved_count}/{self.SAMPLES_PER_SUBJECT}")
                    
            except Exception as e:
                print(f"[MMLULoader] FAILED to load {subject}: {e}")

        random.shuffle(all_selected_data)

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            for entry in all_selected_data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"[MMLULoader] Successfully saved {len(all_selected_data)} questions to {self.file_path}")
    
    def generate_dataset(self):
        self._generate_dataset()

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
    test_cases = loader.generate_dataset()