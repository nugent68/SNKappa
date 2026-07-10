#!/usr/bin/env python
"""Quick probe of cosmoDC2 parquet content (run on Perlmutter)."""
import re
import numpy as np
import pyarrow.parquet as pq

PATH = ("/global/cfs/cdirs/lsst/shared/xgal/cosmoDC2/"
        "cosmoDC2_v1.1.4_parquet/cosmoDC2_v1.1.4_image_healpix10066.parquet")

f = pq.ParquetFile(PATH)
cols = f.schema_arrow.names
print("mag cols:", [c for c in cols if re.search(r"mag(_true)?_[iz]_lsst$", c)])
print("halo cols:", [c for c in cols if "halo" in c])

t = f.read_row_group(0, columns=["ra", "dec", "redshift_true", "stellar_mass",
                                 "halo_mass", "is_central", "convergence",
                                 "mag_z_lsst"]).to_pandas()
print(f"rowgroup0: {len(t)} rows")
print("ra range", t.ra.min(), t.ra.max(), "| dec", t.dec.min(), t.dec.max())
print("z range", t.redshift_true.min(), round(t.redshift_true.max(), 3))
sel = (t.mag_z_lsst <= 22.5) & (t.redshift_true < 1.4)
print("z<=22.5 & zt<1.4:", int(sel.sum()))
print("kappa percentiles(src 0.9<z<1.1):",
      np.percentile(t.convergence[(t.redshift_true > 0.9)
                                  & (t.redshift_true < 1.1)],
                    [2, 16, 50, 84, 98]).round(4))
cl = t[(t.is_central) & (t.halo_mass > 6e13) & (t.redshift_true < 1.15)]
print("clusters (central, Mh>6e13, z<1.15) in rowgroup:", len(cl))
print("stellar_mass units check: max logM* =",
      round(np.log10(t.stellar_mass.max()), 2))
print("halo_mass units check: max logMh =",
      round(np.log10(t.halo_mass.max()), 2))
