#!/usr/bin/env python3
"""Cross-calibration sample: galaxies FB fitted successfully, spanning
mass and redshift, to measure the FB<->Prospector offset that a
two-engine catalog would otherwise hide."""
import csv, glob, random

HDR = "ls_id,lensed,status,n_filters,runtime_s".split(",")
BASE = "/pscratch/sd/n/nugent/lens"
N = 60


def rows(f):
    with open(f) as fh:
        first = fh.readline(); fh.seek(0)
        if first.startswith("ls_id"):
            yield from csv.DictReader(fh)
        else:
            for line in fh:
                p = line.rstrip("\n").split(",")
                if len(p) >= 5:
                    yield dict(zip(HDR, p[:5]))


fb = {}
for f in glob.glob(f"{BASE}/summaries_zfix/summary_zfix_*.csv"):
    for r in rows(f):
        if r.get("status") == "ok" and r.get("logmass_p50"):
            try:
                fb[r["ls_id"]] = float(r["logmass_p50"])
            except ValueError:
                pass

tg = {r["ls_id"]: r for r in csv.DictReader(open(f"{BASE}/fb_targets_all.csv"))}
have = [i for i in fb if i in tg]
# stratify by FB mass so the offset can be checked for mass dependence
have.sort(key=lambda i: fb[i])
step = max(len(have) // N, 1)
sel = have[::step][:N]
with open(f"{BASE}/fb_targets_xcal.csv", "w", newline="") as fo:
    fn = list(next(iter(tg.values())).keys())
    wr = csv.DictWriter(fo, fieldnames=fn); wr.writeheader()
    for i in sel:
        wr.writerow(tg[i])
print(f"xcal sample: {len(sel)} galaxies, FB logM* "
      f"{fb[sel[0]]:.2f} to {fb[sel[-1]]:.2f}")
