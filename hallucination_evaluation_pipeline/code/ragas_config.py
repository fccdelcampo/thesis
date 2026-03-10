import os
import sys
import time
import random

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from llm_config import ENCODER_MODEL_NAME, ENCODER_MODEL_NAME as _LLM_MODEL, USE_TOGETHER_API, get_openai_client, BASE_URL, API_KEY, log_llm_event, retry_with_exponential_backoff
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.llm import OllamaLLM
import threading
from openai import RateLimitError, APITimeoutError, APIConnectionError
from openai import OpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, APIError, APIStatusError


# RAGAS 0.4.x requires wrapped LLM/Embeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Model name imported from the central config
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from llm_config import MODEL_NAME, RAGAS_MODEL_NAME
# EMBED_MODEL_NAME = "nomic-embed-text"

# Add aliases for RAGAS

import os
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler

# Ensure your config variables are imported
from llm_config import RAGAS_MODEL_NAME, EMBED_MODEL, USE_TOGETHER_API, API_KEY, USE_OPENAI_API

import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks import AsyncCallbackHandler

# I recommend changing your config flag from USE_TOGETHER_API to USE_OPENAI_API for clarity
from llm_config import RAGAS_MODEL_NAME, USE_OPENAI_API 

class RagasDebugCallback(AsyncCallbackHandler): # <-- Inherit from AsyncCallbackHandler
    """
    Async LangChain callback to safely log RAGAS parallel activity to debug.txt.
    """
    # FIX: Add 'async' to the method definition
    async def on_chat_model_start(self, serialized, messages, **kwargs):
        try:
            with open("debug.txt", "a", encoding="utf-8") as f:
                f.write("\n--- [DEBUG] RAGAS JUDGE PROMPT ---\n")
                for batch in messages:
                    for msg in batch:
                        f.write(f"[{msg.type.upper()}]: {msg.content}\n")
        except Exception as e: 
            print(f"Logger Error: {e}")
            
    # FIX: Add 'async' to the method definition
    async def on_llm_end(self, response, **kwargs):
        try:
            with open("debug.txt", "a", encoding="utf-8") as f:
                text = response.generations[0][0].text if response.generations else ""
                f.write(f"\n=== [DEBUG] RAGAS JUDGE RESPONSE ===\n{text}\n")
        except Exception: pass

# =================================================================
# THE INTERCEPTOR CLASSES (Bypasses Pydantic strictness)
# =================================================================

class InterceptingChatOpenAI(ChatOpenAI):
    async def agenerate(self, messages, stop=None, callbacks=None, **kwargs):
        # print("\n" + "🔥"*20 + " ASYNC PROMPT INTERCEPTED (OPENAI) " + "🔥"*20)
        # for batch in messages:
        #     for msg in batch:
        #         print(f"[{msg.type.upper()}]:\n{msg.content}\n")
        # print("🔥"*62 + "\n")
        return await super().agenerate(messages, stop=stop, callbacks=callbacks, **kwargs)

class InterceptingChatOllama(ChatOllama):
    async def agenerate(self, messages, stop=None, callbacks=None, **kwargs):
        # print("\n" + "🔥"*20 + " ASYNC PROMPT INTERCEPTED (OLLAMA) " + "🔥"*20)
        # for batch in messages:
        #     for msg in batch:
        #         print(f"[{msg.type.upper()}]:\n{msg.content}\n")
        # print("🔥"*62 + "\n")
        return await super().agenerate(messages, stop=stop, callbacks=callbacks, **kwargs)

# =================================================================

def get_ragas_llm():
    """Returns a pure LangChain model for RAGAS evaluation with a hardcoded prompt interceptor."""

    if USE_OPENAI_API:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("CRITICAL: OPENAI_API_KEY environment variable is missing!")

        # Use our custom subclass instead of the standard ChatOpenAI
        llm = InterceptingChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=openai_key,     
            max_retries=5 
        )
    else:
        # Use our custom subclass instead of the standard ChatOllama
        llm = InterceptingChatOllama(
            model=RAGAS_MODEL_NAME,
            temperature=0.0,
            num_ctx=4096
        )
        raise ValueError("ChatOllama being used, not OpenAI.")
        
    return LangchainLLMWrapper(llm)

import os
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from llm_config import USE_TOGETHER_API

def get_ragas_embeddings():
    """Returns pure LangChain embeddings for RAGAS evaluation."""
    if USE_TOGETHER_API: # Or whatever flag you use to trigger external embeddings
        # 1. Safely extract ONLY the Together API key
        together_key = os.environ.get("TOGETHER_API_KEY")
        if not together_key:
            raise ValueError("CRITICAL: TOGETHER_API_KEY environment variable is missing!")
            
        # 2. Pass it explicitly to the api_key parameter
        embeddings = OpenAIEmbeddings(
            base_url="https://api.together.xyz/v1",
            api_key=together_key, # LangChain will lock onto this key for this instance
            model="intfloat/multilingual-e5-large-instruct",
            max_retries=5
        )
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
    return LangchainEmbeddingsWrapper(embeddings)

class TogetherAIEmbeddings(Embedder):
        """Embedder that calls the Together AI OpenAI-compatible embeddings endpoint."""
        def embed_query(self, text: str):
            from llm_config import EMBED_MODEL
            _client = get_openai_client()
            start_time = time.perf_counter()
            
            # Wrap with exponential backoff
            invoke_func = retry_with_exponential_backoff(_client.embeddings.create)
            
            response = invoke_func(
                model=EMBED_MODEL,
                input=text,
            )
            duration = time.perf_counter() - start_time
            log_llm_event("EMBED_QUERY", duration=duration)
            return response.data[0].embedding

        async def async_embed_chunks(self, texts: list[str]) -> list[list[float]]:
            """Batch embed a list of texts using the Together AI API."""
            from llm_config import EMBED_MODEL
            import asyncio
            _client = get_openai_client()
            loop = asyncio.get_event_loop()
            
            def _do_batch():
                # The openai client can take a list of strings
                start_time = time.perf_counter()
                
                # Wrap with exponential backoff
                invoke_func = retry_with_exponential_backoff(_client.embeddings.create)
                
                resp = invoke_func(model=EMBED_MODEL, input=texts)
                duration = time.perf_counter() - start_time
                log_llm_event(f"EMBED_BATCH_{len(texts)}", duration=duration)
                return [d.embedding for d in resp.data]
            
            return await loop.run_in_executor(None, _do_batch)
        
from langchain_ollama import ChatOllama # Use the official LangChain version
from langchain_core.outputs import LLMResult, Generation

from langchain_ollama import ChatOllama
from pydantic import Field, ConfigDict
from typing import Dict, Any

class DebugOllamaLLM(ChatOllama):
    """Wrapper to clean input prompts and log raw LLM responses to debug.txt."""
    
    # 1. Declare custom fields so Pydantic knows they exist (This fixes the error)
    model_name: str = Field(default="")
    model_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Optional: Tell Pydantic not to crash if you add other custom variables later
    model_config = ConfigDict(extra="allow")

    def __init__(self, model_name: str, model_params: dict = None, **kwargs):
        params = model_params or {}
        options = params.get("options", {})
        
        # 2. Pass everything correctly to the Pydantic parent class
        super().__init__(
            model=model_name,           # LangChain uses 'model' natively
            model_name=model_name,      # Assigns your custom field
            model_params=params,        # Assigns your custom field
            temperature=options.get("temperature", 0),
            num_ctx=options.get("num_ctx", 4096),
            **kwargs
        )
    
    def __init__(self, model_name, model_params=None, **kwargs):
        # Map your custom 'model_name' to LangChain's 'model'
        params = model_params or {}
        options = params.get("options", {})
        super().__init__(
            model=model_name,
            temperature=options.get("temperature", 0),
            num_ctx=options.get("num_ctx", 4096),
            **kwargs
        )
        self.model_name = model_name # Keep for your internal logic

    # --- Keep your existing helper methods ---
    def _log_to_file(self, message: str): ...
    def _clean_response_text(self, text: str) -> str: ...
    def _clean_prompt(self, text: str) -> str: ...
    def _clean_input_data(self, input_data): ...

    # RAGAS 0.4.x specifically calls this or agenerate
    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """Directly satisfies the RAGAS wrapper requirement."""
        return await self.agenerate(prompts, stop=stop, callbacks=callbacks, **kwargs)

    # --- Your Together AI Routing ---
    def invoke(self, input, *args, **kwargs):
        if USE_TOGETHER_API:
            # Your existing _openai_invoke logic
            return self._openai_invoke(input, **kwargs)
        
        # Log and clean before passing to standard LangChain invoke
        clean_input = self._clean_input_data(input)
        self._log_to_file(f"\n--- [DEBUG] SENT PROMPT ---\n{clean_input}")
        
        response = super().invoke(clean_input, *args, **kwargs)
        
        response.content = self._clean_response_text(response.content)
        self._log_to_file(f"=== [DEBUG] RESPONSE ===\n{response.content}")
        return response

def retry_with_exponential_backoff(
    func,
    max_retries=5,
    initial_sleep=1,
    backoff_factor=2,
    jitter=True
):
    """Retries a function with exponential backoff for LLM API calls."""
    def wrapper(*args, **kwargs):
        sleep_time = initial_sleep
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError, APIError, APIStatusError) as e:
                # Also check for specific status codes if it's an APIStatusError
                if isinstance(e, APIStatusError) and e.status_code in [401, 403, 404]:
                    # These are usually not retryable (Auth/Permission/NotFound)
                    raise e

                last_exception = e
                if attempt == max_retries:
                    break
                
                # Calculate sleep duration
                actual_sleep = sleep_time
                if jitter:
                    actual_sleep += random.uniform(0, 1)
                
                log_llm_event(f"RETRY_BACKOFF_ATTEMPT_{attempt+1}", error=f"Waiting {actual_sleep:.2f}s after error: {type(e).__name__}")
                print(f"  > [LLM BACKOFF] Attempt {attempt+1} failed ({type(e).__name__}). Retrying in {actual_sleep:.2f}s...", flush=True)
                
                time.sleep(actual_sleep)
                sleep_time *= backoff_factor
            except Exception as e:
                # For non-retriable exceptions, raise immediately
                raise e
        
        raise last_exception
    return wrapper


def initialize_models():
    llm_judge = DebugOllamaLLM(
        model_name=RAGAS_MODEL_NAME,
        model_params={
            "keep_alive": "20m",
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "format": "json",
                "repeat_penalty": 1.2,
                "seed": 42
            }
        }
    )

    original_invoke = llm_judge.invoke

    # 2. Define the retry rules using Tenacity decorators
    @retry(
        # Wait exponentially: 2s, 4s, 8s, 16s... up to a max of 60 seconds per wait
        wait=wait_exponential(multiplier=1, min=2, max=60),
        
        # Give up completely after 8 failed attempts to prevent infinite hangs
        stop=stop_after_attempt(8),
        
        # ONLY retry on network or rate limit errors (don't retry on bad prompts)
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        
        reraise=True # If it fails 8 times, raise the error so you know it died
    )
    def robust_invoke(*args, **kwargs):
        print("DEBUG PROLOG-GRAPHRAG:Sending chunk to LLM...")
        return original_invoke(*args, **kwargs)

    # 3. Apply the monkey-patch to your LLM instance
    llm_judge.invoke = robust_invoke

    if USE_TOGETHER_API:
        from llm_config import EMBED_MODEL
        embedder = TogetherAIEmbeddings()
        # print(f"Together AI embedder loaded ({EMBED_MODEL}).")
    else:
        from llm_config import EMBED_MODEL
        embedder = OllamaEmbeddings(model=EMBED_MODEL)
        print("Ollama embedder loaded.")

    return llm_judge, embedder
