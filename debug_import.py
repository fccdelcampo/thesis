import sys
import os

# Mimic eval_runner.py path setup
pipeline_dir = os.path.join(os.getcwd(), 'prolog_graphrag_pipeline')
sys.path.append(pipeline_dir)

print(f"sys.path: {sys.path}")

try:
    import graphrag
    print(f"graphrag file: {graphrag.__file__}")
    print(f"graphrag path: {graphrag.__path__}")
except ImportError as e:
    print(f"Could not import graphrag: {e}")

try:
    from graphrag import config
    print(f"config file: {config.__file__}")
    print(f"Has PROMPT_TEMPLATE: {hasattr(config, 'PROMPT_TEMPLATE')}")
except ImportError as e:
    print(f"Could not import graphrag.config: {e}")
except Exception as e:
    print(f"Error accessing config: {e}")
