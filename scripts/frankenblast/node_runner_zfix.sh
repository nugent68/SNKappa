#!/bin/bash
# Production node runner: 8-band FrankenBlast in ZFIX mode (redshift
# conditioned from the SNKappa catalog: DESI spec-z where available,
# else the DECaLS photo-z the kappa pipeline itself uses).
# Probe result: 31/31 galaxies complete, median 300 s, zero timeouts --
# including the 7 that never finished in 3.9 h under the z-free model.
source /pscratch/sd/n/nugent/lens/env.sh
export OMP_NUM_THREADS=2 TORCH_NUM_THREADS=2
export KFB_TMAX_PER_ITER=60
cd /pscratch/sd/n/nugent/lens/frankenblast-host
cat "$SBIPP_PHOT_ROOT"/sbi_phot_zfix_GPD2W_global.h5 \
    "$SBIPP_ROOT"/SBI_model_zfix_GPD2W_global.pt > /dev/null 2>&1
CSV=${KFB_CSV:-/pscratch/sd/n/nugent/lens/fb_targets_all.csv}
OUT=/pscratch/sd/n/nugent/lens/summaries_zfix
mkdir -p "$OUT"
P=${SLURM_PROCID:-0}; NN=${SLURM_NNODES:-1}; WPN=${WPN:-64}
NROWS=$(( $(wc -l < "$CSV") - 1 ))
PER_NODE=$(( (NROWS + NN - 1) / NN ))
NODE_START=$(( P * PER_NODE ))
NODE_END=$(( NODE_START + PER_NODE ))
[ "$NODE_END" -gt "$NROWS" ] && NODE_END=$NROWS
echo "node $P: rows $NODE_START-$NODE_END host=$(hostname)"
worker() {
  local w=$1
  local sumcsv="$OUT/summary_zfix_n${P}_w${w}.csv"
  local i=$(( NODE_START + w ))
  while [ "$i" -lt "$NODE_END" ]; do
    timeout -k 15 1500 python run_kappa_zfix.py "$CSV" "$i" "$((i+1))" "$sumcsv" \
      >> "$OUT/worker_zfix_n${P}_w${w}.log" 2>&1
    rc=$?
    if [ "$rc" -ge 124 ]; then
      lsid=$(sed -n "$((i+2))p" "$CSV" | cut -d, -f1)
      echo "${lsid},False,fail: hard-timeout,0,1500.0" >> "$sumcsv"
    fi
    i=$(( i + WPN ))
  done
}
for w in $(seq 0 $((WPN - 1))); do worker "$w" & done
wait
echo "node $P done"
