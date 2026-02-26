from copy import deepcopy
import signal
import threading

# Prolog code generation benefits from a code-tuned model
PROLOG_MODEL = 'llama3.1:8b-instruct-q4_K_M'
# Natural language tasks (explanation, close-check, final synthesis) stay on llama3.2
NL_MODEL = 'llama3.1:8b-instruct-q4_K_M'
OLLAMA_TIMEOUT = 600  # seconds per LLM call (10 minutes)

# ── KV-cache warmup ────────────────────────────────────────────────────────
# Ollama reuses its KV cache when the prefix of a chat request matches a
# previously seen prefix.  Because the GraphRAG step runs the same model just
# before Prolog generation, it evicts the Prolog prefix from the cache.
# warmup_prolog_model() re-primes the cache (system prompt + few-shots) once
# per question so that all retry attempts within that question are fast.
_kv_cache_warmed = False

def warmup_prolog_model():
    """Prime Ollama's KV cache with the static few-shot prefix (once per question)."""
    global _kv_cache_warmed
    if _kv_cache_warmed:
        return
    import ollama
    try:
        from .prolog_config import GENERATOR_LLM_MESSAGES
    except ImportError:
        from prolog_graphrag_pipeline.prolog.prolog_config import GENERATOR_LLM_MESSAGES
    try:
        print("[Prolog Warmup] Priming KV cache with few-shot prefix...", flush=True)
        ollama.chat(
            model=PROLOG_MODEL,
            messages=GENERATOR_LLM_MESSAGES + [
                {'role': 'user', 'content': 'Now, process the following:\nContext: A is true.\nUser Question: Is A true?\n'}
            ],
            options={'temperature': 0, 'keep_alive': '20m', 'num_ctx': 4096},
        )
        _kv_cache_warmed = True
        print("[Prolog Warmup] KV cache primed.", flush=True)
    except Exception as e:
        print(f"[Prolog Warmup] Non-fatal warmup failure: {e}", flush=True)

def reset_kv_cache_flag():
    """Call this before each new question so warmup re-runs after GraphRAG evicts the cache."""
    global _kv_cache_warmed
    _kv_cache_warmed = False

def reset_ollama_model():
    """Force-unload the model from Ollama, then verify it reloads successfully."""
    import ollama
    try:
        print("[Ollama Reset] Unloading prolog model to recover from hang...", flush=True)
        # Setting keep_alive to 0 with a dummy generate forces immediate unload
        ollama.generate(model=PROLOG_MODEL, prompt="", keep_alive=0)
        print("[Ollama Reset] Model unloaded.", flush=True)
    except Exception as e:
        print(f"[Ollama Reset] Warning during unload: {e}", flush=True)

    # Verify the model reloads successfully with a simple ping
    try:
        print("[Ollama Reset] Verifying model reloads...", flush=True)
        test_response = ollama.chat(
            model=PROLOG_MODEL,
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
    
    # Select model based on flag: code-tuned for Prolog, NL model for everything else
    if flag == "prolog":
        model = PROLOG_MODEL
        llm_messages = GENERATOR_LLM_MESSAGES + [{'role': 'user', 'content': prompt}]
        temperature = 0
    elif flag == "explanation":
        model = NL_MODEL
        llm_messages = EXPLAINER_LLM_MESSAGES + [{'role': 'user', 'content': prompt}]
        temperature = 0.2
    elif flag == "q":
        model = NL_MODEL
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
                model=model,
                messages=llm_messages,
                options={
                    'temperature': temperature,
                    'keep_alive': '20m',
                    'num_ctx': 4096,
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