import json
import os

class HotpotQALoader:
    def __init__(self, file_path="test-cases/hotpotqa.jsonl"):
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
                    
                    # HotpotQA Logic: Flatten context [['Title', ['s1', 's2']]] -> ['Title: s1', 'Title: s2']
                    raw_context = item.get('context', [])
                    flattened_context = []
                    for entry in raw_context:
                        if len(entry) == 2:
                            title, sentences = entry
                            for sent in sentences:
                                flattened_context.append(f"{title}: {sent}")
                    
                    data.append({
                        "dataset": "hotpotqa",
                        "question": item['question'],
                        "formatted_input": item['question'],
                        "ground_truth": item.get('answer', ''),
                        "ground_truth_context": flattened_context
                    })
                except json.JSONDecodeError:
                    continue
        return data
