#!/bin/bash
# setup.sh — Install Python dependencies for the replication package.
# Run once before executing run_paper.py or code/all_code_python.py.
#
# Usage:
#   bash setup.sh
#
# Requires: Python 3.9+ and pip

pip install -r requirements.txt
