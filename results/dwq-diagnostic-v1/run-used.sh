#!/bin/sh
set -eu
cd /home/imad/GitHub/personal/mellum-mlx
.venv/bin/python -u scripts/refine_dwq.py diagnostic --student artifacts/mellum2.1-mixed-C-g64 --destination artifacts/mellum2.1-dwq-diagnostic-1e7 --output results/dwq-diagnostic-v1/train.json > results/dwq-diagnostic-v1/train.log 2>&1
.venv/bin/python -u scripts/refine_dwq.py verify --student artifacts/mellum2.1-dwq-diagnostic-1e7 --output results/dwq-diagnostic-v1/reload.json > results/dwq-diagnostic-v1/reload.log 2>&1
.venv/bin/python scripts/validate.py artifacts/mellum2.1-dwq-diagnostic-1e7 --receipt results/dwq-diagnostic-v1/structure.json > results/dwq-diagnostic-v1/structure.log 2>&1
.venv/bin/python -u scripts/nonexpert.py integrity --model artifacts/mellum2.1-dwq-diagnostic-1e7 --output results/dwq-diagnostic-v1/integrity.json > results/dwq-diagnostic-v1/integrity.log 2>&1
