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

def run_pipeline(question: str):
    result = generate(question)
    return result


if __name__ == "__main__":
    result = run_pipeline("What is the meaning of life in one sentence.")
    print(f"RESULT\n{result}\n\n")