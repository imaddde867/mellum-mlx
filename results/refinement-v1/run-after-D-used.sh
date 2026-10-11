#!/bin/sh
set -eu
cd /home/imad/GitHub/personal/mellum-mlx
while [ ! -f results/refinement-v1/screen-D.json ]; do sleep 2; done
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0,'scripts')
from nonexpert import write_new
r=Path('results/refinement-v1')
d=json.loads((r/'screen-D.json').read_text())
c=json.loads(Path('results/nonexpert-v1/screen-C.json').read_text())
assert d['status']=='complete'
label='D' if d['passed']>c['passed'] or (d['passed']==c['passed'] and d['completion_tokens']<=c['completion_tokens'] and d['generation_s']<c['generation_s']) else 'C'
write_new(r/'selection.json', {'calibration_control':label,'criterion':json.loads((r/'predeclared.json').read_text())['selection'], 'D':{k:d[k] for k in ['passed','generation_s','completion_tokens','weight_bytes']}, 'C':{k:c[k] for k in ['passed','generation_s','completion_tokens','weight_bytes']}, 'decision_scope':'Development selection for one calibration pass; no promotion.'})
PY
CONTROL=$(.venv/bin/python -c 'import json; print(json.load(open("results/refinement-v1/selection.json"))["calibration_control"])')
.venv/bin/python scripts/refine_dwq.py targets --output results/refinement-v1/targets.json > results/refinement-v1/targets.log 2>&1
.venv/bin/python scripts/refine_dwq.py train --student "artifacts/mellum2.1-mixed-$CONTROL-g64" --destination "artifacts/mellum2.1-refinement-dwq" --output results/refinement-v1/train.json > results/refinement-v1/train.log 2>&1
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
