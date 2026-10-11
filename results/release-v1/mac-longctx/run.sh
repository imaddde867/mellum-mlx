#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
OUT=results/release-v1/mac-longctx
for ctx in 65536 130816; do
  echo "$ctx start $(date -u +%FT%TZ) adapter=$(pmset -g ac | awk '/Wattage/{print $3}') pressure=$(sysctl -n kern.memorystatus_vm_pressure_level) $(sysctl -n vm.swapusage | awk '{print "swap_used="$6}')" >> $OUT/system.log
  .venv/bin/python scripts/benchmark.py artifacts/mellum2.1-nvfp4 --context $ctx --max-tokens 256 --trials 1 --output $OUT/nvfp4-$ctx.json > /dev/null
  echo "$ctx end   $(date -u +%FT%TZ) pressure=$(sysctl -n kern.memorystatus_vm_pressure_level) $(sysctl -n vm.swapusage | awk '{print "swap_used="$6}')" >> $OUT/system.log
done
