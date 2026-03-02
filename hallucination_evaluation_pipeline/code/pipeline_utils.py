import sys, os
import requests
from langchain_community.chat_models import ChatOllama

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from llm_config import MODEL_NAME

def unload_ollama_model():
    """Force-unload the model from VRAM so RAGAS can use it without OOM."""
    try:
        requests.post("http://localhost:11434/api/generate", json={
            "model": MODEL_NAME,
            "prompt": "",
            "keep_alive": 0
        }, timeout=10)
        print("  [VRAM] Model unloaded successfully.")
    except Exception as e:
        print(f"  [VRAM] Warning: Could not unload model: {e}")

class CleanOutputChatOllama(ChatOllama):
    """Wrapper that strips non-JSON characters from LLM responses so RAGAS can parse them.
    
    Overrides _generate (which invoke/generate call internally) to ensure
    all code paths produce clean JSON output.
    """
    
    @staticmethod
    def _clean_json(text: str) -> str:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            return text[start:end + 1]
        # Try array format too (some RAGAS prompts expect arrays)
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and start < end:
            return text[start:end + 1]
        return text

    def _clean_result(self, result):
        """Clean all message contents in a ChatResult."""
        if hasattr(result, 'generations'):
            for gen_list in result.generations:
                for gen in gen_list if isinstance(gen_list, list) else [gen_list]:
                    if hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                        gen.message.content = self._clean_json(gen.message.content)
                    if hasattr(gen, 'text'):
                        gen.text = self._clean_json(gen.text)
        return result

    def _generate(self, *args, **kwargs):
        result = super()._generate(*args, **kwargs)
        # ChatResult has .generations list
        if hasattr(result, 'generations'):
            for gen in result.generations:
                if hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                    gen.message.content = self._clean_json(gen.message.content)
                if hasattr(gen, 'text'):
                    gen.text = self._clean_json(gen.text)
        return result

    async def _agenerate(self, *args, **kwargs):
        result = await super()._agenerate(*args, **kwargs)
        if hasattr(result, 'generations'):
            for gen in result.generations:
                if hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                    gen.message.content = self._clean_json(gen.message.content)
                if hasattr(gen, 'text'):
                    gen.text = self._clean_json(gen.text)
        return result

    def invoke(self, *args, **kwargs):
        result = super().invoke(*args, **kwargs)
        if hasattr(result, 'content'):
            result.content = self._clean_json(result.content)
        return result

    async def ainvoke(self, *args, **kwargs):
        result = await super().ainvoke(*args, **kwargs)
        if hasattr(result, 'content'):
            result.content = self._clean_json(result.content)
        return result
