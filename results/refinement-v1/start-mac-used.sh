#!/bin/sh
set -eu
cd /Users/imadeddine/Documents/GitHub/personal/mellum-mlx
MAC=/Users/imadeddine/mellum-mlx-test-20261009
cp "$MAC/scripts/validate.py" "$MAC/work/refinement-validate-before.py"
cp scripts/validate.py "$MAC/scripts/validate.py"
python3 - "$MAC" <<'PY'
import hashlib,sys
from pathlib import Path
root=Path(sys.argv[1])
for name in ['benchmark.py','score.py','validate.py']:
    assert hashlib.sha256((Path('scripts')/name).read_bytes()).digest()==hashlib.sha256((root/'scripts'/name).read_bytes()).digest(),name
assert not (root/'results/refinement-v1/mac-system-snapshots.json').exists()
assert not (root/'results/refinement-v1/mac-gate.log').exists()
PY
cd "$MAC"
/usr/bin/caffeinate -i .venv/bin/python -u scripts/refinement-mac-checks.py > results/refinement-v1/mac-gate.log 2>&1
