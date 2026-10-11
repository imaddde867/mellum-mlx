#!/bin/sh
set -eu
cd /home/imad/GitHub/personal/mellum-mlx
CONTROL=C
.venv/bin/python scripts/refine_dwq.py train --student "artifacts/mellum2.1-mixed-$CONTROL-g64" --destination "artifacts/mellum2.1-refinement-dwq" --output results/refinement-v1/train-checkpoint.json > results/refinement-v1/train-checkpoint.log 2>&1
cp artifacts/mellum2.1-refinement-dwq/conversion.json results/refinement-v1/conversion-DWQ.json
.venv/bin/python scripts/validate.py artifacts/mellum2.1-refinement-dwq --receipt results/refinement-v1/structure-DWQ.json > results/refinement-v1/structure-DWQ.log 2>&1
.venv/bin/python scripts/nonexpert.py integrity --model artifacts/mellum2.1-refinement-dwq --output results/refinement-v1/integrity-DWQ.json > results/refinement-v1/integrity-DWQ.log 2>&1
.venv/bin/python scripts/nonexpert.py screen --label DWQ --model artifacts/mellum2.1-refinement-dwq --output results/refinement-v1/screen-DWQ.json > results/refinement-v1/screen-DWQ.log 2>&1
cp results/nonexpert-v1/integrity-C.json results/refinement-v1/integrity-C.json
.venv/bin/python scripts/nonexpert.py screen --label C --model artifacts/mellum2.1-mixed-C-g64 --output results/refinement-v1/screen-C.json > results/refinement-v1/screen-C.log 2>&1
.venv/bin/python scripts/nonexpert.py screen --label mxfp4 --model work/mxfp4 --output results/refinement-v1/screen-mxfp4.json > results/refinement-v1/screen-mxfp4.log 2>&1
for pair in "$CONTROL artifacts/mellum2.1-mixed-$CONTROL-g64" 'DWQ artifacts/mellum2.1-refinement-dwq' 'mxfp4 work/mxfp4'; do
  set -- $pair
  .venv/bin/python -c 'import sys; from pathlib import Path; sys.path.insert(0,"scripts"); sys.path.insert(0,"work/runtime-code"); import optimization as o; o.DATA=Path("work/heldout"); o.RESULTS=Path("results/refinement-v1"); o.main()' evaluate --split heldout --model "$2" --output "results/refinement-v1/heldout-$1.json" > "results/refinement-v1/heldout-$1.log" 2>&1
done
