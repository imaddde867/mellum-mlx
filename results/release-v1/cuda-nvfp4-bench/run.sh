#!/bin/sh
set -eu
cd ~/GitHub/personal/mellum-mlx
for round in 1 2; do
  for ctx in 1024 4096 16384; do
    for m in "mxfp4 work/mxfp4" "nvfp4 /home/imad/mellum-nvfp4-evidence/nvfp4-model" "nvfp4b /home/imad/mellum-nvfp4-evidence/nvfp4-model" "mxfp4b work/mxfp4"; do
      set -- $m
      .venv/bin/python scripts/benchmark.py "$2" --context $ctx --max-tokens 256 --trials 3 --output /home/imad/mellum-nvfp4-evidence/cuda-bench/$1-r$round-$ctx.json > /dev/null
    done
  done
done
echo done > /home/imad/mellum-nvfp4-evidence/cuda-bench/DONE
