import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent
print(ROOT)

# RESULTS_DIR = ROOT / "hallucination_evaluation_results"
sys.path.append(str(ROOT))
# Define target folder inside top-level directory
# IO_DIR = ROOT / "hallucination_evaluation_results"

try:
    from .llm import generate
except ImportError:
    from llm import generate

def run_pipeline(user_question: str, flag: str = "", sample_mode: bool = False, include_conclusion: bool = True):
    result = generate(user_question)
    return {
        "answer": result['text_answer'] if result else "Error",
        "contexts": [],
        "logprobs": result.get('logprobs') if result else None
    }


if __name__ == "__main__":
    result = run_pipeline("What is the meaning of life in one sentence.")
    print(f"RESULT\n{result}\n\n")
    
    print(f"TEXT ANSWER\n{result['answer']}\n\n")
    print(f"LOGPROBS\n{result['logprobs']}\n\n")