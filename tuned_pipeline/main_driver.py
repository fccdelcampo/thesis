import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent
print(ROOT)

# RESULTS_DIR = ROOT / "hallucination-evaluation-results"
sys.path.append(str(ROOT))
# Define target folder inside top-level directory
# IO_DIR = ROOT / "hallucination-evaluation-results"

from llm import generate

def run_pipeline(user_question: str):
    return generate(user_question)

result = run_pipeline("What is the meaning of life in one sentence.")
print(f"RESULT\n{result}\n\n")

print(f"TEXT ANSWER\n{result["text_answer"]}\n\n")
print(f"LOGPROBS\n{result["logprobs"]}\n\n")