import os
import sys

# Make `controllers.*` and sim/*.py importable exactly as the live nodes import them.
SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sim"))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)
