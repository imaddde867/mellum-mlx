#!/bin/sh
# Use a fresh checkout with cached pinned models; existing evidence is never overwritten.
set -eu
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
STAGE=${1:-all}
RESULTS=${2:-results/nonexpert-v1}
case "$STAGE" in all|screen) ;; *) echo 'Use all or screen' >&2; exit 1;; esac
mkdir -p "$RESULTS"
if [ "$STAGE" = all ] && [ -n "$(ls -A "$RESULTS")" ]; then
  echo "Refusing existing evidence in $RESULTS; use an empty result directory" >&2
  exit 1
fi
for label in bf16 4bit mxfp4 A B C; do
  for suffix in json raw.jsonl memory.json server.log log; do
    if [ -e "$RESULTS/screen-$label.$suffix" ]; then
      echo "Refusing existing screen evidence in $RESULTS" >&2
      exit 1
    fi
  done
done
if [ -e "$RESULTS/preflight.json" ]; then
  echo "Refusing existing preflight evidence in $RESULTS" >&2
  exit 1
fi
if [ "$STAGE" = all ]; then
  for recipe in A B C; do
    "$PY" scripts/convert.py --recipe "$recipe" --estimate-only > "$RESULTS/estimate-$recipe.json"
  done
  for recipe in A B C; do
    "$PY" scripts/convert.py --recipe "$recipe" > "$RESULTS/convert-$recipe.log" 2>&1
    cp "artifacts/mellum2.1-mixed-$recipe-g64/conversion.json" "$RESULTS/conversion-$recipe.json"
    "$PY" scripts/validate.py "artifacts/mellum2.1-mixed-$recipe-g64" --receipt "$RESULTS/structure-$recipe.json" > "$RESULTS/structure-$recipe.log" 2>&1
    "$PY" scripts/nonexpert.py integrity --model "artifacts/mellum2.1-mixed-$recipe-g64" --output "$RESULTS/integrity-$recipe.json" > "$RESULTS/integrity-$recipe.log" 2>&1
  done
fi
"$PY" - "$RESULTS" <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
from convert import REVISION, sha256
from nonexpert import frozen, sandbox, write_new
from validate import check_hashes
root = Path('artifacts/mellum2.1-affine-4bit-g64')
manifest = json.loads((root/'conversion.json').read_text())
assert manifest['source_revision'] == REVISION
check_hashes(Path('work/source'), manifest['source_sha256'])
check_hashes(root, manifest['artifact_sha256'])
external = Path('work/mxfp4')
revision = '4ce0d28df07dfa4ed28077e5b718e866be2cf4cf'
assert sha256(external/'model.safetensors') == 'b6224c8ea3e32190acc1500ce513d39263be257c6c71a10a31e832c045d2cb9a'
metadata = {}
for p in external.glob('.cache/huggingface/download/*.metadata'):
    lines = p.read_text().splitlines()
    assert lines[0] == revision, f'Comparator revision mismatch: {p}'
    name = p.name.removesuffix('.metadata')
    metadata[name] = {'download_revision':lines[0], 'sha256':sha256(external/name)}
assert 'config.json' in metadata and 'tokenizer.json' in metadata
frozen()
case = {'entry_point':'add', 'checks':'assert add(2,3)==5\nassert add(-1,1)==0'}
assert sandbox(case, 'def add(a,b):\n return a+b')['passed']
assert not sandbox(case, 'def add(a,b):\n return a-b')['passed']
write_new(Path(sys.argv[1])/'preflight.json', {'source_revision':REVISION,
    'source_sha256':manifest['source_sha256'], 'native4_sha256':manifest['artifact_sha256'],
    'external_revision':revision, 'external_files':metadata, 'sandbox_positive_negative':'passed'})
PY
for pair in 'bf16 work/source' '4bit artifacts/mellum2.1-affine-4bit-g64' 'mxfp4 work/mxfp4' 'A artifacts/mellum2.1-mixed-A-g64' 'B artifacts/mellum2.1-mixed-B-g64' 'C artifacts/mellum2.1-mixed-C-g64'; do
  set -- $pair
  "$PY" scripts/nonexpert.py screen --label "$1" --model "$2" --output "$RESULTS/screen-$1.json" > "$RESULTS/screen-$1.log" 2>&1
done
