import sys

# from pathlib import Path

# ROOT = Path(__file__).resolve().parent
# print(ROOT)

# RESULTS_DIR = ROOT / "hallucination_evaluation_results"
# sys.path.append(str(ROOT))
# Define target folder inside top-level directory
# IO_DIR = ROOT / "hallucination_evaluation_results"
try:
    from .config import LLM_MESSAGES
except ImportError:
    try:
        from config import LLM_MESSAGES
    except ImportError:
        # Fallback if config is in the same directory but not found (e.g. shadowed)
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from config import LLM_MESSAGES
import ollama

def generate(user_question: str):
    try:
        answer = dict()
        response = ollama.chat(
                model='llama3.1:8b-instruct-q4_K_M',
                messages=LLM_MESSAGES + [{"role": "user", "content": user_question}],
                options={
                    'temperature': 0.4,
                    'keep_alive': 0,
                    'top_k': 50,
                    'top_p': 0.9
                },
                logprobs=True     
            )
        answer['text_answer'] = response['message']['content']
        answer['logprobs'] = response.get('logprobs', None)
        return answer
    except Exception as e:
        print(f"Error during Ollama interaction: {e}")
        return None
        