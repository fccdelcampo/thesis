import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Sys Path: {sys.path}")

try:
    import ragas
    print(f"Ragas Version: {ragas.__version__}")
    print(f"Ragas Path: {ragas.__file__}")
except ImportError:
    print("Ragas package not found.")

# Check for namespaces if installed
import site
print(f"Site Packages: {site.getsitepackages()}")
