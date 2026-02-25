from copy import deepcopy
import signal
import threading

MODEL_NAME = 'llama3.1:8b-instruct-q4_K_M'
OLLAMA_TIMEOUT = 600  # seconds per LLM call (10 minutes)

def reset_ollama_model():
    """Force-unload the model from Ollama, then verify it reloads successfully."""
    import ollama
    try:
        print("[Ollama Reset] Unloading model to recover from hang...", flush=True)
        # Setting keep_alive to 0 with a dummy generate forces immediate unload
        ollama.generate(model=MODEL_NAME, prompt="", keep_alive=0)
        print("[Ollama Reset] Model unloaded.", flush=True)
    except Exception as e:
        print(f"[Ollama Reset] Warning during unload: {e}", flush=True)

    # Verify the model reloads successfully with a simple ping
    try:
        print("[Ollama Reset] Verifying model reloads...", flush=True)
        test_response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': 'Say OK'}],
            options={'temperature': 0, 'keep_alive': '20m'},
        )
        reply = test_response.get('message', {}).get('content', '').strip()
        print(f"[Ollama Reset] Model reloaded successfully. Ping reply: {reply[:50]}", flush=True)
    except Exception as e:
        print(f"[Ollama Reset] WARNING: Model failed to reload: {e}", flush=True)

def generate(prompt: str, flag: str) -> dict:
    if flag not in ["prolog", "explanation", "q"]:
        raise ValueError(f"Flag {flag} is invalid. Only 'prolog', 'explanation', and 'q' are accepted.")
    
    return generate_response(prompt=prompt, flag=flag)

    
def generate_response(prompt: str, flag: str) -> dict:
    from .prolog_config import GENERATOR_LLM_MESSAGES, EXPLAINER_LLM_MESSAGES
    import ollama
    
    answer = dict()
    
    if flag == "prolog":
        llm_messages = GENERATOR_LLM_MESSAGES + [{'role': 'user', 'content': prompt}]
        temperature = 0
    elif flag == "explanation":
        llm_messages = EXPLAINER_LLM_MESSAGES + [{'role': 'user', 'content': prompt}]
        temperature = 0.2
    elif flag == "q":
        llm_messages = [{'role': 'user', 'content': prompt}]
        temperature = 0
    else:
        raise ValueError(f"Flag {flag} is invalid. Only 'prolog', 'explanation', and 'q' are accepted.")

    # Use a thread to enforce a hard timeout on ollama.chat()
    result_holder = [None]
    error_holder = [None]

    def _call_ollama():
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=llm_messages,
                options={
                    'temperature': temperature,
                    'keep_alive': '20m'
                },
            )
            result_holder[0] = response
        except Exception as e:
            error_holder[0] = e

    thread = threading.Thread(target=_call_ollama, daemon=True)
    thread.start()
    thread.join(timeout=OLLAMA_TIMEOUT)

    if thread.is_alive():
        # Ollama is hung — reset and raise
        print(f"[Ollama TIMEOUT] LLM call exceeded {OLLAMA_TIMEOUT}s. Resetting model...", flush=True)
        reset_ollama_model()
        raise TimeoutError(f"Ollama LLM call timed out after {OLLAMA_TIMEOUT}s")

    if error_holder[0]:
        print(f"Error during Ollama interaction: {error_holder[0]}")
        return None

    response = result_holder[0]
    if response is None:
        return None

    try:
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