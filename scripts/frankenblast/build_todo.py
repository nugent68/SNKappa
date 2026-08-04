#!/usr/bin/env python3
"""Write the still-unfitted subset of the FB target list.

Scans every zfix summary (tolerating files whose first row is a
headerless timeout line -- the shell fallback in the node runners
creates those), and emits fb_targets_todo.csv with the galaxies that
have no successful fit yet.
"""
import csv, glob, sys

HDR = "ls_id,lensed,status,n_filters,runtime_s".split(",")
BASE = "/pscratch/sd/n/nugent/lens"


def rows(f):
    with open(f) as fh:
        first = fh.readline()
        fh.seek(0)
        if first.startswith("ls_id"):
            yield from csv.DictReader(fh)
        else:
            for line in fh:
                p = line.rstrip("\n").split(",")
                if len(p) >= 5:
                    yield dict(zip(HDR, p[:5]))


ok = set()
for f in glob.glob(f"{BASE}/summaries_zfix/summary_zfix_*.csv"):
    for r in rows(f):
        if r.get("status") == "ok":
            ok.add(r["ls_id"])

src = f"{BASE}/fb_targets_all.csv"
out = f"{BASE}/fb_targets_todo.csv"
n = 0
with open(src) as fin, open(out, "w", newline="") as fout:
    rd = csv.DictReader(fin)
    wr = csv.DictWriter(fout, fieldnames=rd.fieldnames)
    wr.writeheader()
    for r in rd:
        if r["ls_id"] not in ok:
            wr.writerow(r)
            n += 1
print(f"fitted {len(ok)} | todo {n} -> {out}")
