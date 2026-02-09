import json
import os

class MMLULoader:
    def __init__(self, file_path="test-cases/mmlu_selected.jsonl"):
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
                    # MMLU Logic: Format question + options
                    options = item.get('options', [])
                    formatted_options = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                    formatted_input = f"{item['question']}\n{formatted_options}"
                    
                    # Ground Truth is the index (0-3), convert to A-D
                    gt_idx = item.get('answer', 0)
                    ground_truth = chr(65 + gt_idx) if isinstance(gt_idx, int) else str(gt_idx)

                    data.append({
                        "dataset": "mmlu",
                        "question": item['question'],
                        "formatted_input": formatted_input,
                        "ground_truth": ground_truth,
                        "ground_truth_context": None # Impossible for MMLU
                    })
                except json.JSONDecodeError:
                    continue
        return data
