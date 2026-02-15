from .config import LLM_SYSTEM_PROMPT, LLM_MESSAGES
import re
import ollama

from .prompt_reconstructor import reconstruct_prompt

def generate(user_question: str, retrieved_context: str | None, explainer_output: str | None, flag: str, sample_mode: bool = False) -> dict | list[dict]:
    final_prompt = reconstruct_prompt(user_question=user_question, retrieved_context=retrieved_context, explainer_output=explainer_output, flag=flag)
    if sample_mode:
        sequences = []
        for i in range(5):
            try:
                answer = dict()
                response = ollama.chat(
                        model='llama3.1:8b-instruct-q4_K_M',
                        messages=LLM_MESSAGES + [{"role": "user", "content": final_prompt}],
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
        