#!/bin/bash
# Tier-2 runner: direct prospector-alpha/dynesty for galaxies the SBI
# emulator cannot fit even with the redshift conditioned. Prospector
# has no training distribution, so the out-of-distribution failure
# mode does not exist for it -- these galaxies are simply data.
# zred is pinned from the SNKappa catalog (PROS_FIXZ default).
# CSV is chosen by KFB_CSV so the same runner serves both the residual
# tier and the FB-overlap cross-calibration sample.
source /pscratch/sd/n/nugent/lens/env.sh
export OMP_NUM_THREADS=2 TORCH_NUM_THREADS=1
export PROS_FIXZ=1 PROS_MAXCALL=${PROS_MAXCALL:-600000}
cd /pscratch/sd/n/nugent/lens/frankenblast-host
CSV=${KFB_CSV:-/pscratch/sd/n/nugent/lens/fb_targets_todo.csv}
OUT=${PROS_OUT:-/pscratch/sd/n/nugent/lens/summaries_pros_tier2}
TAG=${PROS_TAG:-t2}
mkdir -p "$OUT"
P=${SLURM_PROCID:-0}; NN=${SLURM_NNODES:-1}; WPN=${WPN:-48}
NROWS=$(( $(wc -l < "$CSV") - 1 ))
PER_NODE=$(( (NROWS + NN - 1) / NN ))
NODE_START=$(( P * PER_NODE )); NODE_END=$(( NODE_START + PER_NODE ))
[ "$NODE_END" -gt "$NROWS" ] && NODE_END=$NROWS
echo "pros-$TAG node $P: rows $NODE_START-$NODE_END host=$(hostname)"
worker() {
  local w=$1
  local sumcsv="$OUT/summary_pros_${TAG}_n${P}_w${w}.csv"
  local i=$(( NODE_START + w ))
  while [ "$i" -lt "$NODE_END" ]; do
    timeout -k 15 ${PROS_CAP:-7000} python run_prospector_catalog.py \
      "$CSV" "$i" "$((i+1))" "$sumcsv" \
      >> "$OUT/worker_pros_${TAG}_n${P}_w${w}.log" 2>&1
    rc=$?
    if [ "$rc" -ge 124 ]; then
      lsid=$(sed -n "$((i+2))p" "$CSV" | cut -d, -f1)
      echo "${lsid},fail: hard-timeout,0,${PROS_CAP:-7000}" >> "$sumcsv"
    fi
    i=$(( i + WPN ))
  done
}
for w in $(seq 0 $((WPN - 1))); do worker "$w" & done
wait
echo "pros-$TAG node $P done"
