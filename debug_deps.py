
import sys
import traceback

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

print("\n--- Testing scipy.stats ---")
try:
    from scipy.stats import kruskal, mannwhitneyu, rankdata
    print("SUCCESS: scipy.stats imported")
except Exception:
    traceback.print_exc()

print("\n--- Testing statsmodels ---")
try:
    from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    print("SUCCESS: statsmodels imported")
except Exception:
    traceback.print_exc()

print("\n--- Testing scikit-posthocs ---")
try:
    from scikit_posthocs import posthoc_dunn
    print("SUCCESS: scikit-posthocs imported")
except Exception:
    traceback.print_exc()
