import os
import json
import random
from datasets import load_dataset

# Configuration
OUTPUT_FILE = 'test-cases/mmlu.jsonl'
SAMPLES_PER_SUBJECT = 10  # 10 subjects * 10 samples = 100 total
RANDOM_SEED = 42          # reproducibility

# educational subjects with focus on reasoning
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

def generate_mmlu_dataset():
    print(f"Starting MMLU generation for {len(TARGET_SUBJECTS)} subjects...")
    print(f"Strategy: Randomly sampling {SAMPLES_PER_SUBJECT} questions per subject.\n")
    
    all_selected_data = []
    
    for subject in TARGET_SUBJECTS:
        try:
            print(f"Fetching subject: {subject}...")
            # Removed trust_remote_code=True as it is no longer needed for cais/mmlu
            ds = load_dataset("cais/mmlu", subject, split="test")
            
            # SHUFFLE to avoid "Chapter 1 bias"
            shuffled_ds = ds.shuffle(seed=RANDOM_SEED)
            
            # Select the first N from the SHUFFLED list
            n_samples = min(len(ds), SAMPLES_PER_SUBJECT)
            samples = shuffled_ds.select(range(n_samples))
            
            for item in samples:
                # Map integer answer (0-3) to Letter (A-D)
                answer_key = ["A", "B", "C", "D"]
                correct_char = answer_key[item['answer']]
                
                # Standardize
                entry = {
                    "question": item['question'],
                    "options": item['choices'],
                    "answer": correct_char,
                    "subject": subject
                }
                all_selected_data.append(entry)
                
        except Exception as e:
            print(f"FAILED to load {subject}: {e}")

    random.shuffle(all_selected_data)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_selected_data:
            f.write(json.dumps(entry) + '\n')

    print(f"\nSuccess! Saved {len(all_selected_data)} questions to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_mmlu_dataset()