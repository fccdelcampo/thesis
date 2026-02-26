import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ollama
from prolog_graphrag_pipeline.prolog.prolog_config import GENERATOR_LLM_MESSAGES

MODELS = [
    "llama3.2:3b-instruct-q8_0",
    "qwen2.5-coder:3b-instruct-q8_0",
    "llama3.1:8b-instruct-q4_K_M"
]
JUDGE_MODEL = "llama3.1:8b-instruct-q4_K_M"

TEST_CASES = [
    {
        "question": "Is Maria the grandmother of Dan?",
        "context": "Maria is the mother of Chloe. Chloe is the mother of Dan."
    },
    {
        "question": "Who does Vito believe he should keep closer?",
        "context": "Tomasinno is the friend of Vito. Solozzo is the enemy of Vito. Vito believes that he should keep his enemies closer than his friends."
    },
    {
        "question": "If the grass is contaminated, are the snakes affected?",
        "context": "In a food web, grass is a producer. Grasshoppers eat grass. Frogs eat grasshoppers. Snakes eat frogs."
    }
]

def generate_prolog(model, question, context):
    prompt = f"Now, process the following:\nContext:\n{context}\n\nUser Question:\n{question}\n"
    messages = GENERATOR_LLM_MESSAGES + [{'role': 'user', 'content': prompt}]
    
    response = ollama.chat(
        model=model,
        messages=messages,
        options={'temperature': 0}
    )
    return response['message']['content']

def evaluate_with_judge(question, context, generated_prolog):
    prompt = f"""You are an expert Prolog evaluator.
Evaluate the following raw LLM response containing Prolog code for accuracy and correctness given the context and question.
Check if it properly uses <database> and <query> tags.
Check if the logic maps correctly to the context.

Context: {context}
Question: {question}

Generated Response:
{generated_prolog}

Score the generated Prolog from 1 to 10 based on syntax, logic, and alignment with the context.
Provide a brief explanation, then end with "Score: [X]/10".
"""
    try:
        response = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to judge model: {e}"

def main():
    for i, test in enumerate(TEST_CASES):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}")
        print(f"Context: {test['context']}")
        print(f"Question: {test['question']}")
        print(f"{'='*60}")
        
        for model in MODELS:
            print(f"\n>>> Generating with {model}...")
            try:
                result = generate_prolog(model, test["question"], test["context"])
                print("--- Generated Output ---")
                print(result)
                print("------------------------")
                
                print(f"\n>>> Judging with {JUDGE_MODEL}...")
                judge_feedback = evaluate_with_judge(test["question"], test["context"], result)
                print("--- Judge Feedback ---")
                print(judge_feedback)
                print("----------------------\n")
            except Exception as e:
                print(f"Error evaluating model {model}: {e}")

if __name__ == "__main__":
    main()
