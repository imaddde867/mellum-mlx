#!/bin/sh
set -eu
cd /Users/imadeddine/Documents/GitHub/personal/mellum-mlx
MAC=/Users/imadeddine/mellum-mlx-test-20261009
PART="$MAC/work/refinement-interrupted-transfer"
mkdir "$PART"
mv "$MAC/artifacts/mellum2.1-refinement-dwq/model-00001-of-00002.safetensors" "$PART/"
cp "$MAC/artifacts/mellum2.1-affine-4bit-g64/model-00001-of-00002.safetensors" "$MAC/artifacts/mellum2.1-refinement-dwq/"
rsync -r --partial --stats --copy-dest="$MAC/artifacts/mellum2.1-affine-4bit-g64" 5090-1:/home/imad/GitHub/personal/mellum-mlx/artifacts/mellum2.1-refinement-dwq/ "$MAC/artifacts/mellum2.1-refinement-dwq/"
scp 5090-1:/home/imad/GitHub/personal/mellum-mlx/results/refinement-v1/mac-selection.json "$MAC/results/refinement-v1/"
cp results/refinement-v1/runtime-code/mac-checks.py "$MAC/scripts/refinement-mac-checks.py"
cp results/refinement-v1/mac-rule.json results/refinement-v1/editor-verification.json results/refinement-v1/mac-workload-environment.json "$MAC/results/refinement-v1/"
python3 - "$MAC" <<'PY'
import hashlib,sys
from pathlib import Path
root=Path(sys.argv[1])
for name in ['benchmark.py','score.py','validate.py']:
    assert hashlib.sha256((Path('scripts')/name).read_bytes()).digest()==hashlib.sha256((root/'scripts'/name).read_bytes()).digest(),name
PY
cd "$MAC"
/usr/bin/caffeinate -i .venv/bin/python -u scripts/refinement-mac-checks.py > results/refinement-v1/mac-gate.log 2>&1
