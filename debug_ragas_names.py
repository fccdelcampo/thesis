import sys
# Use the known working path to ragas
try:
    import ragas.metrics
    print(f"ragas.metrics available: {True}")
    print(f"Names in ragas.metrics: {dir(ragas.metrics)}")
except Exception as e:
    print(f"Error importing ragas.metrics: {e}")
