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

class OpenBookQALoader:
    """
    Loader for the OpenBookQA dataset (allenai/openbookqa).
    """
    
    # --- Configuration ---
    FILE_PATH = 'test_cases/openbookqa.jsonl'
    SAMPLE_SIZE = 100
    RANDOM_SEED = 42

    def __init__(self, sample_size=None):
        if sample_size is not None:
             self.SAMPLE_SIZE = sample_size
        self.file_path = self.FILE_PATH

    def ensure_data_exists(self):
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            return

        print(f"[OpenBookQALoader] File {self.file_path} missing or empty. Generating now...")
        self._generate_dataset()

    def _generate_dataset(self):
        random.seed(self.RANDOM_SEED)
        all_selected_data = []

        print(f"[OpenBookQALoader] Fetching 'allenai/openbookqa' from Hugging Face...")
        
        try:
            # We use the main config and validation split
            ds = load_dataset("allenai/openbookqa", "main", split="validation")
            
            total_available = len(ds)
            print(f"[OpenBookQALoader] Total available: {total_available}. Goal: {self.SAMPLE_SIZE} approved.")
            
            indices = list(range(total_available))
            random.shuffle(indices)
            
            approved_count = 0
            
            for idx in indices:
                if approved_count >= self.SAMPLE_SIZE:
                     break
                 
                item = ds[idx]
                
                question_stem = item.get('question_stem', '')
                choices_text = item.get('choices', {}).get('text', [])
                choices_label = item.get('choices', {}).get('label', [])
                answer_key = item.get('answerKey', '')

                # Format options for the LLM to approve and evaluate
                formatted_options_approve = " ".join([f"{lbl}) {txt}" for lbl, txt in zip(choices_label, choices_text)])
                approval_prompt = f"{question_stem}? Choices: {formatted_options_approve}"
                
                # Start logging to file
                os.makedirs("test_cases/logs", exist_ok=True)
                with open("test_cases/logs/openbookqa_approval.log", "a", encoding="utf-8") as log_file:
                    log_file.write(f"Question: {question_stem}\n")
                    log_file.flush()

                    if approve(approval_prompt):
                        entry = {
                            "dataset" : "openbookqa",
                            "question": question_stem,
                            "options": choices_text,
                            "labels": choices_label,
                            "answer": answer_key
                        }
                        
                        all_selected_data.append(entry)
                        approved_count += 1
                        items_left = self.SAMPLE_SIZE - approved_count
                        log_file.write(f"[Approved!] (Needed: {items_left}, Approved: {approved_count})\n")
                        log_file.flush()
                        print(f"   [+] Approved. Count: {approved_count}/{self.SAMPLE_SIZE}")
                    else:
                        items_left = self.SAMPLE_SIZE - approved_count
                        log_file.write(f"[Rejected!] (Needed: {items_left}, Approved: {approved_count})\n")
                        log_file.flush()

        except Exception as e:
            print(f"[OpenBookQALoader] FAILED to load: {e}")
            import traceback
            traceback.print_exc()

        if all_selected_data:
            os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                for entry in all_selected_data:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"[OpenBookQALoader] Successfully saved {len(all_selected_data)} questions to {self.file_path}")
        else:
            print("[OpenBookQALoader] No data selected/saved.")
        
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
                        
                        options_text = item.get('options', [])
                        options_label = item.get('labels', [])
                        
                        formatted_options = "\n".join([f"{lbl}) {txt}" for lbl, txt in zip(options_label, options_text)])
                        formatted_input = f"{item['question']}\n{formatted_options}"
                        
                        data.append({
                            "dataset": "openbookqa",
                            "question": item['question'],
                            "formatted_input": formatted_input,
                            "ground_truth": item['answer'], 
                            "ground_truth_context": None
                        })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[OpenBookQALoader] Read Error: {e}")
            return []
        return data

if __name__ == "__main__":
    loader = OpenBookQALoader()
    loader.generate_dataset()
