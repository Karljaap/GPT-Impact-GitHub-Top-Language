"""
run_paper.py
------------
Entry point to reproduce all results.
Runs code/all_code.py, which executes all sections sequentially:
  1. Data cleaning & panel construction
  2. Language distribution charts
  3. Programming language trend charts
  4. ChatGPT global availability map
  5. Stata DID / SC / SDID estimations (base model)
  6. Merge control variables + Stata estimations with controls

Requirements:
  pip install -r requirements.txt
  Stata 15+ installed and accessible from the system path
  (or set STATA_EXE inside code/all_code.py)
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
script   = os.path.join(BASE_DIR, "code", "all_code.py")

print("=" * 60)
print("Running full replication pipeline: code/all_code.py")
print("=" * 60)

result = subprocess.run([sys.executable, script], cwd=BASE_DIR)
sys.exit(result.returncode)
