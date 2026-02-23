import ollama
try:
    from .config import LLM_SYSTEM_PROMPT, LLM_MESSAGES
    from .prompt_reconstructor import reconstruct_prompt
except ImportError:
    import sys
    import os
    # Add project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from prolog_graphrag_pipeline.config import LLM_SYSTEM_PROMPT, LLM_MESSAGES
    from prolog_graphrag_pipeline.prompt_reconstructor import reconstruct_prompt

from typing import Optional, Union, List

def generate(user_question: str, retrieved_context: Optional[str], explainer_output: Optional[str], flag: str, sample_mode: bool = False, include_conclusion: bool = True, context_from_prompt: str = "") -> Union[dict, List[dict]]:
    final_prompt = reconstruct_prompt(user_question=user_question, retrieved_context=retrieved_context, explainer_output=explainer_output, flag=flag, context_from_prompt=context_from_prompt)
    if sample_mode:
        sequences = []
        for i in range(5):
            try:
                answer = dict()
                response = ollama.chat(
                        model='llama3.1:8b-instruct-q4_K_M',
                        messages=LLM_MESSAGES + [{"role": "user", "content": final_prompt}],
                        options={
                            'temperature': 0.2,
                            'keep_alive': 0,
                            'top_k': 50,
                            'top_p': 0.9
                        },
                        logprobs=True     
                    )
                answer['text_answer'] = response['message']['content']
                answer['logprobs'] = response.get('logprobs', None)
                sequences.append(answer)
            except Exception as e:
                print(f"Error during Ollama interaction: {e}")
                return None
        return sequences
    else:
        try:
            answer = dict()
            response = ollama.chat(
                    model='llama3.1:8b-instruct-q4_K_M',
                    messages=LLM_MESSAGES + [{"role": "user", "content": final_prompt}],
                    options={
                        'temperature': 0.2,
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
        