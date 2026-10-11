#!/bin/sh
# Clean interleaved NVFP4 vs MXFP4 decode benchmark for publication numbers (~35 min).
set -eu
cd "$(dirname "$0")/.."
W=$(pmset -g ac | awk '/Wattage/{print $3}' | tr -d W)
[ "${W:-0}" -ge 60 ] || { echo "ABORT: charger negotiates ${W:-0} W; need >= 60 W (use the MacBook's own USB-C charger)."; exit 1; }
[ "$(pmset -g | awk '/lowpowermode/{print $2}')" = 0 ] || { echo "ABORT: Low Power Mode is on."; exit 1; }
for f in work/mxfp4/config.json artifacts/mellum2.1-nvfp4/config.json; do [ -f $f ] || { echo "ABORT: missing $f"; exit 1; }; done
OUT=results/release-v1/mac-abba-clean
[ ! -e $OUT ] || { echo "ABORT: $OUT already exists; refusing to overwrite."; exit 1; }
mkdir -p $OUT; L=$OUT/system.log
echo "Charger ${W} W OK. Running ~35 min; don't use the Mac until it prints the result."
M=work/mxfp4; N=artifacts/mellum2.1-nvfp4
run() {
  echo "$1 $3 start $(date -u +%FT%TZ) $(pmset -g batt | sed -n 2p | tr -s ' \t' ' ') adapter=$(pmset -g ac | awk '/Wattage/{print $3}') pressure=$(sysctl -n kern.memorystatus_vm_pressure_level) $(sysctl -n vm.swapusage | awk '{print "swap_used="$6}')" >> $L
  .venv/bin/python scripts/benchmark.py "$2" --context $3 --max-tokens 256 --trials 3 --output "$OUT/$1-$3.json" > /dev/null
  printf '.'
}
for ctx in 1024 4096 16384; do
  rounds=2; [ $ctx = 1024 ] && rounds=3
  r=1
  while [ $r -le $rounds ]; do
    if [ $((r % 2)) = 1 ]; then run mxfp4-r$r-a $M $ctx; run nvfp4-r$r-a $N $ctx; run nvfp4-r$r-b $N $ctx; run mxfp4-r$r-b $M $ctx
    else run nvfp4-r$r-a $N $ctx; run mxfp4-r$r-a $M $ctx; run mxfp4-r$r-b $M $ctx; run nvfp4-r$r-b $N $ctx; fi
    r=$((r + 1))
  done
done
echo "done $(date -u +%FT%TZ)" >> $L
echo; .venv/bin/python results/release-v1/mac-abba/summary.py $OUT
