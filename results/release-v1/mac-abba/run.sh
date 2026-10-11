#!/bin/sh
# Interleaved A-B-B-A decode comparison; both models see the same power/thermal conditions.
set -eu
cd "$(dirname "$0")/.."
OUT=results/release-v1/mac-abba; L=$OUT/system.log
M=work/mxfp4; N=artifacts/mellum2.1-nvfp4
run() { # label model ctx
  echo "$1 $3 start $(date -u +%FT%TZ) $(pmset -g batt | sed -n 2p | tr -s ' \t' ' ') adapter=$(pmset -g ac | awk '/Wattage/{print $3}') pressure=$(sysctl -n kern.memorystatus_vm_pressure_level) $(sysctl -n vm.swapusage | awk '{print "swap_used="$6}')" >> $L
  .venv/bin/python scripts/benchmark.py "$2" --context $3 --max-tokens 256 --trials 3 --output "$OUT/$1-$3.json" > /dev/null
}
for ctx in 1024 4096 16384; do
  rounds=2; [ $ctx = 1024 ] && rounds=3
  r=1
  while [ $r -le $rounds ]; do
    if [ $((r % 2)) = 1 ]; then
      run mxfp4-r$r-a $M $ctx; run nvfp4-r$r-a $N $ctx; run nvfp4-r$r-b $N $ctx; run mxfp4-r$r-b $M $ctx
    else
      run nvfp4-r$r-a $N $ctx; run mxfp4-r$r-a $M $ctx; run mxfp4-r$r-b $M $ctx; run nvfp4-r$r-b $N $ctx
    fi
    r=$((r + 1))
  done
done
echo "done $(date -u +%FT%TZ)" >> $L
