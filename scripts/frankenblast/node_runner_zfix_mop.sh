#!/bin/bash
# Mop-up runner: the galaxies no chunk finished, with a longer
# per-galaxy cap (1700 s, the most a 30-min debug wall allows).
# Writes a header before any fallback timeout line so the summaries
# stay machine-readable (the chunk runners did not).
source /pscratch/sd/n/nugent/lens/env.sh
export OMP_NUM_THREADS=2 TORCH_NUM_THREADS=2
export KFB_TMAX_PER_ITER=60
cd /pscratch/sd/n/nugent/lens/frankenblast-host
CSV=/pscratch/sd/n/nugent/lens/fb_targets_todo.csv
OUT=/pscratch/sd/n/nugent/lens/summaries_zfix
mkdir -p "$OUT"
P=${SLURM_PROCID:-0}; NN=${SLURM_NNODES:-1}; WPN=${WPN:-64}
NROWS=$(( $(wc -l < "$CSV") - 1 ))
PER_NODE=$(( (NROWS + NN - 1) / NN ))
NODE_START=$(( P * PER_NODE )); NODE_END=$(( NODE_START + PER_NODE ))
[ "$NODE_END" -gt "$NROWS" ] && NODE_END=$NROWS
echo "mop node $P: rows $NODE_START-$NODE_END host=$(hostname)"
worker() {
  local w=$1
  local sumcsv="$OUT/summary_zfix_mop_n${P}_w${w}.csv"
  [ -f "$sumcsv" ] || echo "ls_id,lensed,status,n_filters,runtime_s" > "$sumcsv"
  local i=$(( NODE_START + w ))
  while [ "$i" -lt "$NODE_END" ]; do
    timeout -k 10 1700 python run_kappa_zfix.py "$CSV" "$i" "$((i+1))" "$sumcsv" \
      >> "$OUT/worker_zfix_mop_n${P}_w${w}.log" 2>&1
    rc=$?
    if [ "$rc" -ge 124 ]; then
      lsid=$(sed -n "$((i+2))p" "$CSV" | cut -d, -f1)
      echo "${lsid},False,fail: hard-timeout,0,1700.0" >> "$sumcsv"
    fi
    i=$(( i + WPN ))
  done
}
for w in $(seq 0 $((WPN - 1))); do worker "$w" & done
wait
echo "mop node $P done"
