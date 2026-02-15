import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prolog_graphrag_pipeline.graphrag.graphrag_driver import clear_local_data

if __name__ == "__main__":
    print("Preparing to clear local data (Documents and Chunks)...")
    confirmation = input("Are you sure? This cannot be undone. Type 'yes' to proceed: ")
    if confirmation.lower() == 'yes':
        clear_local_data()
    else:
        print("Operation cancelled.")
