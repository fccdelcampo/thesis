import sys
import os

# Ensure project root is on the path so llm_config can be imported
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from llm_config import MODEL_NAME, get_openai_client

try:
    from .config import LLM_SYSTEM_PROMPT, LLM_MESSAGES
    from .prompt_reconstructor import reconstruct_prompt
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from prolog_graphrag_pipeline.config import LLM_SYSTEM_PROMPT, LLM_MESSAGES
    from prolog_graphrag_pipeline.prompt_reconstructor import reconstruct_prompt

from typing import Optional, Union, List

client = get_openai_client()


def generate(user_question: str, retrieved_context: Optional[str], explainer_output: Optional[str], flag: str, sample_mode: bool = False, include_conclusion: bool = True, context_from_prompt: str = "", verified_choice: Optional[str] = None, graphrag_answer: str = "") -> Union[dict, List[dict]]:
    final_prompt = reconstruct_prompt(user_question=user_question, retrieved_context=retrieved_context, explainer_output=explainer_output, flag=flag, context_from_prompt=context_from_prompt, verified_choice=verified_choice, graphrag_answer=graphrag_answer)

    def _do_call_llm():
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=LLM_MESSAGES + [{"role": "user", "content": final_prompt}],
            temperature=0.0,
        )
        answer = dict()
        answer['text_answer'] = response.choices[0].message.content
        answer['logprobs'] = None
        return answer

    def _call_llm():
        from llm_config import retry_with_exponential_backoff
        try:
            func = retry_with_exponential_backoff(_do_call_llm)
            return func()
        except Exception as e:
            err_msg = f"Error during synthesis LLM interaction ({MODEL_NAME}): {e}"
            print(err_msg)
            try:
                with open("debug_synthesis_llm.txt", "a", encoding="utf-8") as f:
                    f.write(err_msg + f"\nPROMPT LENGTH: {len(final_prompt)}\n\n")
            except:
                pass
            raise


    if sample_mode:
        sequences = []
        for i in range(5):
            try:
                sequences.append(_call_llm())
            except Exception as e:
                print(f"Error during LLM interaction ({MODEL_NAME}): {e}")
                return None
        return sequences
    else:
        try:
            return _call_llm()
        except Exception as e:
            print(f"Error during LLM interaction ({MODEL_NAME}): {e}")
            return None