#!/bin/sh
set -eu
cd /Users/imadeddine/Documents/GitHub/personal/mellum-mlx
MAC=/Users/imadeddine/mellum-mlx-test-20261009
while ! ssh 5090-1 'cd /home/imad/GitHub/personal/mellum-mlx && .venv/bin/python -c '\''import json,pathlib; r=pathlib.Path("results/refinement-v1"); assert all((r/f"heldout-{x}.json").exists() and json.load((r/f"heldout-{x}.json").open())["status"]=="complete" for x in ["C","DWQ","mxfp4"])'\'' ' >/dev/null 2>&1; do sleep 10; done
[ ! -e "$MAC/artifacts/mellum2.1-refinement-dwq" ]
[ ! -e "$MAC/results/refinement-v1" ]
mkdir -p "$MAC/results/refinement-v1"
scp -r 5090-1:/home/imad/GitHub/personal/mellum-mlx/artifacts/mellum2.1-refinement-dwq "$MAC/artifacts/"
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
