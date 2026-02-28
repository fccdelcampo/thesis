import os
import json
import random
import math
from datasets import load_dataset
from test_case_chooser import approve

class MMLULoader:
    """
    A unified loader that auto-generates the MMLU dataset if it's missing,
    then loads and formats it for the GraphRAG pipeline.
    """
    
    # --- Configuration ---
    FILE_PATH = 'test_cases/mmlu.jsonl'
    SAMPLE_SIZE = 100
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

    def __init__(self, sample_size=None):
        self.file_path = self.FILE_PATH
        if sample_size is not None:
            self.SAMPLE_SIZE = sample_size

    def ensure_data_exists(self):
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            return

        print(f"[MMLULoader] File {self.file_path} missing or empty. Generating now...")
        self._generate_dataset()

    def _generate_dataset(self, append=False):
        random.seed(self.RANDOM_SEED)
        all_selected_data = []

        print(f"[MMLULoader] Fetching {len(self.TARGET_SUBJECTS)} subjects from Hugging Face...")
        
        target_per_subject = math.ceil(self.SAMPLE_SIZE / len(self.TARGET_SUBJECTS))
        print(f"[MMLULoader] Total Goal: {self.SAMPLE_SIZE}, Target per subject: {target_per_subject}")

        for subject in self.TARGET_SUBJECTS:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test")
                
                total_in_subject = len(ds)
                print(f"[MMLULoader] Subject '{subject}': Total available {total_in_subject}. Goal: {target_per_subject} approved.")
                
                subject_approved_count = 0
                seen_indices = set()
                
                while subject_approved_count < target_per_subject:
                    # Stop if we've seen everyone
                    if len(seen_indices) >= total_in_subject:
                        print(f"[MMLULoader] Exhausted subject '{subject}'! Found {subject_approved_count} approved.")
                        break

                    needed = target_per_subject - subject_approved_count
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
                        if subject_approved_count >= target_per_subject:
                            break

                        # Map 0-3 to A-D
                        answer_key = ["A", "B", "C", "D"]
                        correct_char = answer_key[item['answer']]
                        
                        entry = {
                            "question": item['question'],
                            "options": item['choices'],
                            "correct_answer": correct_char,
                            "subject": subject
                        }
                        # Start logging to file
                        os.makedirs("test_cases/logs", exist_ok=True)
                        with open("test_cases/logs/mmlu_approval.log", "a", encoding="utf-8") as log_file:
                            log_file.write(f"Question: {item['question']}\n")
                            log_file.flush()
                            
                            if approve(item['question'] + "? Choices: " + " ".join(item['choices'])):
                                all_selected_data.append(entry)
                                subject_approved_count += 1
                                currently_approved = len(all_selected_data)
                                items_left = self.SAMPLE_SIZE - currently_approved
                                log_file.write(f"[Approved!] (Needed: {items_left}, Approved: {currently_approved})\n")
                                log_file.flush()
                                print(f"   [+] Approved ({subject}). Subject count: {subject_approved_count}/{target_per_subject}")
                            else:
                                currently_approved = len(all_selected_data)
                                items_left = self.SAMPLE_SIZE - currently_approved
                                log_file.write(f"[Rejected!] (Needed: {items_left}, Approved: {currently_approved})\n")
                                log_file.flush()
                    
            except Exception as e:
                print(f"[MMLULoader] FAILED to load {subject}: {e}")

        random.shuffle(all_selected_data)

        mode = 'a' if append else 'w'
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, mode, encoding='utf-8') as f:
            for entry in all_selected_data:
                f.write(json.dumps(entry) + '\n')
        
        print(f"[MMLULoader] Successfully saved {len(all_selected_data)} questions to {self.file_path} (Mode: {mode})")
    
    def generate_dataset(self, append=False):
        self._generate_dataset(append=append)

    def load_data(self):
        self.ensure_data_exists()
        
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
    import argparse
    parser = argparse.ArgumentParser(description="MMLU Dataset Loader")
    parser.add_argument("--subject", type=str, help="Specific subject to load (e.g. formal_logic)")
    parser.add_argument("--n", type=int, help="Number of questions to approve from the subject")
    parser.add_argument("--append", action="store_true", help="Append to the output file instead of overwriting")
    args = parser.parse_args()

    loader = MMLULoader(sample_size=args.n if args.n else None)
    
    if args.subject:
        loader.TARGET_SUBJECTS = [args.subject]
        
    loader.generate_dataset(append=args.append)