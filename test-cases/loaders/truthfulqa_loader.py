import json
import os

class TruthfulQALoader:
    def __init__(self, file_path="test-cases/truthfulqa.jsonl"):
        self.file_path = file_path

    def load_data(self):
        data = []
        if not os.path.exists(self.file_path):
            print(f"Warning: {self.file_path} not found.")
            return []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # TruthfulQA Logic: Simple question pass-through
                    data.append({
                        "dataset": "truthfulqa",
                        "question": item['question'],
                        "formatted_input": item['question'],
                        "ground_truth": item.get('best_answer', ''),
                        "ground_truth_context": None # Impossible for TruthfulQA
                    })
                except json.JSONDecodeError:
                    continue
        return data
