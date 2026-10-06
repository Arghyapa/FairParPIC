#!/usr/bin/env bash
# Runs the full experimental pipeline of the paper.
# Requires the three datasets in data/ (see data/README.md).
set -euo pipefail
mkdir -p results

# Real-data comparison (Tables 2 and 3): n = 3,000, five seeds, delta = 0.05.
# k and k_nn are the elbow-selected values reported in the paper
# (drop --k / --n-neighbors to re-run the elbow selection).
python main_pipeline.py --dataset diabetes --k 7 --n-neighbors 12 --out results/diabetes.json
python main_pipeline.py --dataset census   --k 7 --n-neighbors 10 --out results/census.json
python main_pipeline.py --dataset bank     --k 7 --n-neighbors 12 --out results/bank.json
