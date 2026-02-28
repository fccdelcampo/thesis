import sys
import os
import json

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prolog_graphrag_pipeline.main_driver import run_pipeline as run_prolog_graphrag
from hallucination_evaluation_pipeline.code.mmlu import load_test_cases, format_mmlu_question, extract_answer_letter, score_answer

def main():
    test_cases = load_test_cases(limit=21, start_idx=0)
    for i, test_case in enumerate(test_cases, start=0):
        formatted_input = format_mmlu_question(test_case)
        correct_answer = test_case.get("correct_answer", "")
        raw_question = test_case.get("question", "")
        
        print("="*80)
        print(f"[MMLU #{i}] {raw_question[:120]}...")
        print(f"Correct Answer: [{correct_answer}]")
        print("-" * 80)
        
        try:
            result = run_prolog_graphrag(formatted_input, flag="prolog")
            answer = result.get("answer", "")
            predicted = extract_answer_letter(answer)
            score = score_answer(predicted, correct_answer)
            print(f"Predicted: [{predicted}] | Score: {score}")
            print(f"Full Answer:\n{answer}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
