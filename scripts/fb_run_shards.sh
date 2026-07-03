#!/bin/bash
# Supervise N parallel fb_fit.py shards, then merge results (incl. pilot).
# Usage: bash fb_run_shards.sh <SNKappa_outdir> <frankenblast_dir> <sps_home> [nshards]
set -u
OUTDIR=$1
FBDIR=$2
export SPS_HOME=$3
NS=${4:-10}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

cd "$FBDIR"
pids=()
for i in $(seq 0 $((NS - 1))); do
  .venv-fb/bin/python "$OUTDIR/../../scripts/fb_fit.py" \
    --targets "$OUTDIR/fb_shard${i}.csv" \
    --out "$OUTDIR/fb_results_shard${i}.csv" \
    --training-root sbi_models/sbi_training_sets \
    > "$OUTDIR/fb_shard${i}.log" 2>&1 &
  pids+=($!)
done
echo "launched $NS shards: ${pids[*]}"

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done

# merge shards + pilot
python3 - "$OUTDIR" <<'PYEOF'
import sys, glob
import csv
outdir = sys.argv[1]
rows, header = [], None
for path in sorted(glob.glob(f"{outdir}/fb_results_shard*.csv")) + [f"{outdir}/fb_pilot.csv"]:
    try:
        with open(path) as fh:
            r = list(csv.reader(fh))
    except FileNotFoundError:
        continue
    if not r:
        continue
    if header is None:
        header = r[0]
    rows += [row for row in r[1:] if row and row[0] != "ls_id"]
with open(f"{outdir}/fb_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(header)
    w.writerows(rows)
print(f"merged {len(rows)} fits -> {outdir}/fb_results.csv")
PYEOF
echo "SHARDS DONE (fail=$fail)"
