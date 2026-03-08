import json

def add_test_cases(path):
    test_cases = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            test_cases.append(json.loads(line))
    for i, test_case in enumerate(test_cases):
        test_case['test_case_id'] = i + 1

    with open(path, 'w', encoding='utf-8') as file:
        for test_case in test_cases:
            json.dump(test_case, file)
            file.write('\n')
            
def main():
    ai2_arc_path = "/Users/franciscarlodelcampo/Documents/thesis/test_cases/ai2_arc.jsonl"
    mmlu_path = "/Users/franciscarlodelcampo/Documents/thesis/test_cases/mmlu.jsonl"
    openbookqa_path = "/Users/franciscarlodelcampo/Documents/thesis/test_cases/openbookqa.jsonl"
    metrics_path = "/Users/franciscarlodelcampo/Documents/thesis/test_cases/metrics_test_cases.jsonl"
    add_test_cases(ai2_arc_path)
    add_test_cases(mmlu_path)
    add_test_cases(openbookqa_path)
    add_test_cases(metrics_path)
    
if __name__ == "__main__":
    main()
