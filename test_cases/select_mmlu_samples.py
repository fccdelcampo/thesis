import json
import random
import os

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'mmlu_50_per_sub.jsonl')
    output_path = os.path.join(current_dir, 'mmlu.jsonl')

    print(f"Reading from: {input_path}")
    
    # Read the file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # Parse JSON and group by subject
    subjects_data = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            subject = data.get('subject')
            if subject:
                if subject not in subjects_data:
                    subjects_data[subject] = []
                subjects_data[subject].append(data)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse line {i+1}")

    print(f"Found {len(subjects_data)} subjects: {list(subjects_data.keys())}")

    # Select 10 random questions from each subject
    selected_questions = []
    
    # Check if we have exactly 10 subjects, just for information
    if len(subjects_data) != 10:
        print(f"Note: The file contains {len(subjects_data)} subjects, not exactly 10.")

    for subject, items in subjects_data.items():
        count = len(items)
        if count >= 10:
            selected = random.sample(items, 10)
        else:
            print(f"Warning: Subject '{subject}' has only {count} questions. Selecting all of them.")
            selected = items
        selected_questions.extend(selected)

    # Write to output file
    print(f"Writing {len(selected_questions)} selected questions to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in selected_questions:
            f.write(json.dumps(question) + '\n')
            
    print("Done.")

if __name__ == "__main__":
    main()
