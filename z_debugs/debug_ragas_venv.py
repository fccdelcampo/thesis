import sys
import ragas.metrics

print(f"Ragas metrics dir: {dir(ragas.metrics)}")

try:
    from ragas.metrics import faithfulness
    print("Direct import SUCCESS")
except ImportError:
    print("Direct import FAILED")

try:
    from ragas.metrics.collections import faithfulness
    print("Collections import SUCCESS")
except ImportError:
    print("Collections import FAILED")
