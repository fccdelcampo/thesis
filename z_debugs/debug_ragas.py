import sys
try:
    import ragas
    print(f"Ragas version: {ragas.__version__}")
    print(f"Ragas path: {ragas.__path__}")
except ImportError as e:
    print(f"Could not import ragas: {e}")

print("\n--- Attempting Imports ---")
try:
    from ragas.metrics import faithfulness
    print("Success: from ragas.metrics import faithfulness")
except ImportError as e:
    print(f"Failed: from ragas.metrics import faithfulness ({e})")

try:
    from ragas.metrics.collections import faithfulness
    print("Success: from ragas.metrics.collections import faithfulness")
except ImportError as e:
    print(f"Failed: from ragas.metrics.collections import faithfulness ({e})")

try:
    from ragas import faithfulness
    print("Success: from ragas import faithfulness")
except ImportError as e:
    print(f"Failed: from ragas import faithfulness ({e})")
