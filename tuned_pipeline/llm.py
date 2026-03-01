import sys
import os

# Ensure project root is on the path so llm_config can be imported
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from llm_config import MODEL_NAME, get_openai_client

try:
    from .config import LLM_MESSAGES
except ImportError:
    try:
        from config import LLM_MESSAGES
    except ImportError:
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from config import LLM_MESSAGES

client = get_openai_client()


def generate(user_question: str):
    try:
        answer = dict()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=LLM_MESSAGES + [{"role": "user", "content": user_question}],
            temperature=1.0,
        )
        answer['text_answer'] = response.choices[0].message.content
        # logprobs not universally supported across providers
        answer['logprobs'] = None
        return answer
    except Exception as e:
        print(f"Error during LLM interaction ({MODEL_NAME}): {e}")
        return None