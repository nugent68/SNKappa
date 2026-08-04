#!/bin/bash
# Debug-queue chunk runner: same zfix fitting as node_runner_zfix.sh,
# but bounded to a global row range [KFB_START, KFB_END) so the 6943
# targets can be swept by several short 8-node debug jobs instead of
# waiting on a 2-node regular allocation.
# Per-galaxy cap 900 s (3x the observed median) so a straggler cannot
# hold a worker past the 30-min wall; anything cut off is logged and
# picked up by the mop-up pass.
source /pscratch/sd/n/nugent/lens/env.sh
export OMP_NUM_THREADS=2 TORCH_NUM_THREADS=2
export KFB_TMAX_PER_ITER=60
cd /pscratch/sd/n/nugent/lens/frankenblast-host
CSV=${KFB_CSV:-/pscratch/sd/n/nugent/lens/fb_targets_all.csv}
OUT=/pscratch/sd/n/nugent/lens/summaries_zfix
mkdir -p "$OUT"
P=${SLURM_PROCID:-0}; NN=${SLURM_NNODES:-1}; WPN=${WPN:-64}
NROWS=$(( $(wc -l < "$CSV") - 1 ))
START=${KFB_START:-0}; END=${KFB_END:-$NROWS}
[ "$END" -gt "$NROWS" ] && END=$NROWS
SPAN=$(( END - START ))
PER_NODE=$(( (SPAN + NN - 1) / NN ))
NODE_START=$(( START + P * PER_NODE ))
NODE_END=$(( NODE_START + PER_NODE ))
[ "$NODE_END" -gt "$END" ] && NODE_END=$END
echo "chunk[$START,$END) node $P: rows $NODE_START-$NODE_END host=$(hostname)"
worker() {
  local w=$1
  local sumcsv="$OUT/summary_zfix_c${KFB_TAG}_n${P}_w${w}.csv"
  local i=$(( NODE_START + w ))
  while [ "$i" -lt "$NODE_END" ]; do
    timeout -k 10 900 python run_kappa_zfix.py "$CSV" "$i" "$((i+1))" "$sumcsv" \
      >> "$OUT/worker_zfix_c${KFB_TAG}_n${P}_w${w}.log" 2>&1
    rc=$?
    if [ "$rc" -ge 124 ]; then
      lsid=$(sed -n "$((i+2))p" "$CSV" | cut -d, -f1)
      echo "${lsid},False,fail: hard-timeout,0,900.0" >> "$sumcsv"
    fi
    i=$(( i + WPN ))
  done
}
for w in $(seq 0 $((WPN - 1))); do worker "$w" & done
wait
echo "chunk node $P done"
