"""Outputs: JSON summary, merged LOS catalog (FITS + parquet), provenance."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

from . import __version__


def git_commit() -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def provenance(cfg, tap, report) -> dict:
    import astropy
    import pyvo
    import scipy
    from importlib.metadata import version as pkg_version
    return {
        "snkappa_version": __version__,
        "git_commit": git_commit(),
        "config_hash": cfg.config_hash(),
        "seed": cfg.montecarlo.seed,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_releases": {"ls": cfg.data.ls_release,
                          "desi": cfg.data.desi_release},
        "availability_report": report,
        "query_manifest": tap.manifest,
        "package_versions": {
            "numpy": np.__version__, "scipy": scipy.__version__,
            "astropy": astropy.__version__, "pyvo": pyvo.__version__,
            "colossus": pkg_version("colossus"), "pandas": pd.__version__,
        },
        "acknowledgments": [
            "DESI DR1 (DESI Collaboration et al. 2025, CC BY 4.0)",
            "DESI Legacy Imaging Surveys DR10 (Dey et al. 2019); "
            "https://www.legacysurvey.org/acknowledgment/",
            "Astro Data Lab (CSDC Program of NSF NOIRLab)",
        ],
    }


def write_summary(outdir: Path, summary: dict) -> Path:
    path = outdir / "kappa_ext_summary.json"

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"not serializable: {type(o)}")

    path.write_text(json.dumps(summary, indent=2, default=default))
    return path


def write_catalog(outdir: Path, df: pd.DataFrame, name="los_catalog") -> None:
    """Merged LOS catalog with per-galaxy kappa_i as FITS and parquet."""
    df = df.copy()
    df.to_parquet(outdir / f"{name}.parquet")
    # FITS: cast object columns to str
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
    Table.from_pandas(df).write(outdir / f"{name}.fits", overwrite=True)


def write_samples(outdir: Path, samples: dict) -> None:
    np.savez_compressed(outdir / "kappa_samples.npz", **samples)
