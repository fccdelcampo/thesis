"""
Centralized LLM provider configuration.

Toggle USE_TOGETHER_API to switch every pipeline module between
a local Ollama instance and the Together AI cloud API.

Both providers are accessed through the standard `openai` Python package.
"""

import os
from openai import OpenAI
import httpx
import logging

# Suppress verbose httpx/openai INFO logs (which spam terminal on API retries)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ── Provider Toggle ──────────────────────────────────────────────────────────
# True  → Together AI  (cloud)
# False → Ollama       (local, http://localhost:11434)
USE_TOGETHER_API = True

# ── Model / Endpoint Configuration ──────────────────────────────────────────
if USE_TOGETHER_API:
    BASE_URL = "https://api.together.xyz/v1"
    API_KEY = os.environ.get("TOGETHER_API_KEY", "")
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"
    EMBED_DIM = 1024
    if not API_KEY:
        raise EnvironmentError(
            "TOGETHER_API_KEY environment variable is not set. "
            "Export it before running:  set TOGETHER_API_KEY=your_key"
        )
else:
    BASE_URL = "http://localhost:11434/v1"
    API_KEY = "ollama"          # dummy key required by the openai library
    MODEL_NAME = "llama3:instruct"
    EMBED_MODEL = "llama3:instruct"
    EMBED_DIM = 4096


def get_openai_client() -> OpenAI:
    """Return a configured OpenAI client for the active provider."""
    # Use explicit httpx client with timeout to prevent silent hangs
    http_client = httpx.Client(timeout=60.0)
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, http_client=http_client, max_retries=0)
