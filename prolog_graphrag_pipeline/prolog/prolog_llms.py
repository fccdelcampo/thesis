from copy import deepcopy

def generate(prompt: str, flag: str) -> dict:
    if flag not in ["prolog", "explanation"]:
        raise ValueError(f"Flag {flag} is invalid. Only 'prolog' and 'explanation' are accepted.")
    
    return generate_response(prompt=prompt, flag=flag)
    
def generate_response(prompt: str, flag: str) -> dict:
    from prolog_config import GENERATOR_LLM_MESSAGES, EXPLAINER_LLM_MESSAGES
    import ollama
    
    answer = dict()
    
    if flag == "prolog":
        llm_messages = GENERATOR_LLM_MESSAGES + [{'role': 'user', 'content': prompt}]
        temperature = 0
    elif flag == "explanation":
        llm_messages = EXPLAINER_LLM_MESSAGES + [{'role': 'user', 'content': prompt}]
        temperature = 0.2
    else:
        raise ValueError(f"Flag {flag} is invalid. Only 'prolog' and 'explanation' are accepted.")
    try:
        response = ollama.chat(
            model='llama3.1:8b-instruct-q4_K_M',
            messages=llm_messages,
            options={
                'temperature': temperature,
                'keep_alive': 0
            },
            logprobs=True     
        )
    
        answer['text_answer'] = response['message']['content']
        answer['logprobs'] = response.get('logprobs', None)
        return answer
    except Exception as e:
        print(f"Error during Ollama interaction: {e}")
        return None
    
# import ollama
# response = ollama.chat(
#             model='llama3.1:8b-instruct-q4_K_M',
#             messages=[{'role': 'user', 'content': "What is the Riemann sum definition of the definite integral?"}],
#             options={
#                 'temperature': 0  
#             },
#             logprobs=True     
#         )
    
# print(response['message']['content'])