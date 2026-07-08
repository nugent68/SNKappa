"""NOIRLab Astro Data Lab TAP access: cached queries + STEP-0 availability check.

All data access goes through the public TAP service (anonymous). Every query is
cached to parquet keyed by a hash of the ADQL string, and recorded in a manifest
(query, row count, UTC timestamp, cache status) for auditability.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyvo


class TapClient:
    def __init__(self, tap_url: str, cache_dir: str | Path):
        self.tap_url = tap_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._service: pyvo.dal.TAPService | None = None
        self.manifest: list[dict] = []

    @property
    def service(self) -> pyvo.dal.TAPService:
        if self._service is None:
            self._service = pyvo.dal.TAPService(self.tap_url)
        return self._service

    def _cache_path(self, adql: str) -> Path:
        key = hashlib.sha256(adql.encode()).hexdigest()[:24]
        return self.cache_dir / f"tap_{key}.parquet"

    def query(self, adql: str, label: str = "", maxrec: int = 2_000_000) -> pd.DataFrame:
        """Run an ADQL query with disk caching; sync first, async fallback."""
        path = self._cache_path(adql)
        entry = {
            "label": label,
            "adql": adql,
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if path.exists():
            df = pd.read_parquet(path)
            entry.update(nrows=len(df), cached=True)
            self.manifest.append(entry)
            return df

        result = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pyvo VOTable unit chatter
            last_exc = None
            for attempt, pause in enumerate((0, 45, 120)):
                if pause:
                    time.sleep(pause)
                try:
                    result = self.service.search(adql, maxrec=maxrec)
                    break
                except Exception as exc:  # timeouts / transient 5xx
                    last_exc = exc
                try:
                    # sync failed; try async (also retried with backoff)
                    job = self.service.submit_job(adql, maxrec=maxrec)
                    job.run()
                    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"],
                             timeout=1800.0)
                    job.raise_if_error()
                    result = job.fetch_result()
                    break
                except Exception as exc:
                    last_exc = exc
            if result is None:
                raise RuntimeError(
                    f"TAP query failed after retries: {last_exc}")
        table = result.to_table()
        # object columns (e.g. masked strings) -> plain types for parquet
        df = table.to_pandas()
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)
        df.to_parquet(path)
        entry.update(nrows=len(df), cached=False)
        self.manifest.append(entry)
        return df


# ---------------------------------------------------------------------------
# STEP 0: availability check
# ---------------------------------------------------------------------------

# Columns snkappa needs from each table (superset; i-band absent in the north
# is handled downstream via NaN fluxes, not schema absence).
REQUIRED = {
    "tractor": [
        "ls_id", "ra", "dec", "type", "brick_primary", "maskbits",
        "flux_g", "flux_r", "flux_i", "flux_z",
        "flux_ivar_g", "flux_ivar_r", "flux_ivar_i", "flux_ivar_z",
        "mw_transmission_g", "mw_transmission_r", "mw_transmission_i",
        "mw_transmission_z",
        "fracflux_g", "fracflux_r", "fracflux_z",
        "fracmasked_g", "fracmasked_r", "fracmasked_z",
        "fracin_g", "fracin_r", "fracin_z",
    ],
    "photo_z": [
        "ls_id", "z_phot_median", "z_phot_std",
        "z_phot_l68", "z_phot_u68", "z_phot_l95", "z_phot_u95",
    ],
    "zpix": [
        "targetid", "mean_fiber_ra", "mean_fiber_dec", "z", "zerr", "zwarn",
        "spectype", "zcat_primary", "coadd_fiberstatus",
    ],
}


def _columns(tap: TapClient, table: str) -> set[str]:
    df = tap.query(
        f"SELECT column_name FROM tap_schema.columns WHERE table_name='{table}'",
        label=f"schema:{table}",
    )
    return set(df["column_name"].str.lower())


def availability_report(tap: TapClient, cfg) -> dict:
    """Verify tables/columns against the live TAP_SCHEMA and probe the target field.

    Returns a provenance dict; prints a human-readable report.
    """
    ls = cfg.data.ls_release        # e.g. 'dr10'
    desi = cfg.data.desi_release    # e.g. 'dr1'
    ra, dec = cfg.source.ra_src, cfg.source.dec_src

    report: dict = {"tap_url": tap.tap_url, "checks": {}, "tables": {}}

    schemas = tap.query(
        "SELECT schema_name FROM tap_schema.schemas WHERE schema_name IN "
        f"('ls_{ls}', 'ls_dr9', 'desi_{desi}')",
        label="schema:schemas",
    )["schema_name"].tolist()

    tables = {
        "tractor": f"ls_{ls}.tractor",
        "photo_z": f"ls_{ls}.photo_z",
        "photo_z_fallback": "ls_dr9.photo_z",
        "zpix": f"desi_{desi}.zpix",
    }

    ok = True
    for role, table in tables.items():
        cols = _columns(tap, table)
        exists = len(cols) > 0
        need = set(c.lower() for c in REQUIRED.get(role.replace("_fallback", ""), []))
        missing = sorted(need - cols) if exists else sorted(need)
        report["tables"][role] = {
            "table": table, "exists": exists, "missing_columns": missing,
        }
        if not exists or (missing and role != "photo_z_fallback"):
            ok = ok and (role == "photo_z_fallback")

    # Probe 1: tractor coverage at target
    probe = tap.query(
        f"SELECT COUNT(*) AS n FROM {tables['tractor']} "
        f"WHERE 't'=Q3C_RADIAL_QUERY(ra, dec, {ra}, {dec}, 0.05)",
        label="probe:tractor",
    )
    n_tractor = int(probe["n"].iloc[0])
    report["checks"]["tractor_sources_3arcmin"] = n_tractor

    # Probe 2: DR10 photo-z coverage at target (south-only table!)
    probe = tap.query(
        f"SELECT COUNT(*) AS n FROM {tables['tractor']} t "
        f"JOIN {tables['photo_z']} p ON t.ls_id=p.ls_id "
        f"WHERE 't'=Q3C_RADIAL_QUERY(t.ra, t.dec, {ra}, {dec}, 0.05)",
        label="probe:photo_z",
    )
    n_pz = int(probe["n"].iloc[0])
    report["checks"][f"ls_{ls}_photoz_matches_3arcmin"] = n_pz

    photoz_source = f"ls_{ls}.photo_z"
    if n_pz == 0:
        probe = tap.query(
            "SELECT COUNT(*) AS n FROM ls_dr9.tractor t "
            "JOIN ls_dr9.photo_z p ON t.ls_id=p.ls_id "
            f"WHERE 't'=Q3C_RADIAL_QUERY(t.ra, t.dec, {ra}, {dec}, 0.05)",
            label="probe:photo_z_dr9",
        )
        n_pz9 = int(probe["n"].iloc[0])
        report["checks"]["ls_dr9_photoz_matches_3arcmin"] = n_pz9
        photoz_source = "ls_dr9.photo_z (fallback: target in north, "
        photoz_source += f"ls_{ls}.photo_z is south-only)"
        if n_pz9 == 0:
            ok = False
    report["photoz_source"] = photoz_source

    # Probe 3: DESI spec-z availability at target
    probe = tap.query(
        f"SELECT COUNT(*) AS n FROM {tables['zpix']} "
        f"WHERE 't'=Q3C_RADIAL_QUERY(mean_fiber_ra, mean_fiber_dec, {ra}, {dec}, 0.25) "
        "AND zwarn=0 AND spectype='GALAXY'",
        label="probe:zpix",
    )
    n_desi = int(probe["n"].iloc[0])
    report["checks"]["desi_galaxies_15arcmin"] = n_desi

    report["ok"] = ok and n_tractor > 0
    report["deviations"] = [
        "zpix has no OBJTYPE column; quality cuts use zcat_primary, zwarn=0, "
        "spectype='GALAXY', coadd_fiberstatus=0",
        "photo-z tables serve quantiles only (no full PDF); p(z) reconstructed "
        "as a two-piece Gaussian pinned to median/l68/u68",
    ]
    if "dr9" in photoz_source:
        report["deviations"].append(
            "target field is in the LS north (BASS/MzLS): ls_dr10.photo_z has no "
            "rows there; using ls_dr9.photo_z via identical ls_id, and g-z color "
            "for stellar masses (no i-band in the north)"
        )

    _print_report(report, schemas)
    return report


def _print_report(report: dict, schemas: list[str]) -> None:
    print("=" * 72)
    print("STEP 0 - NOIRLab Data Lab availability report")
    print("=" * 72)
    print(f"TAP endpoint : {report['tap_url']}")
    print(f"Schemas found: {', '.join(sorted(schemas))}")
    for role, info in report["tables"].items():
        status = "OK" if info["exists"] else "MISSING"
        line = f"  {info['table']:24s} [{status}]"
        if info["missing_columns"]:
            line += f"  missing: {', '.join(info['missing_columns'])}"
        print(line)
    for check, val in report["checks"].items():
        print(f"  {check:38s} = {val}")
    print(f"photo-z source: {report['photoz_source']}")
    print("Deviations from nominal spec:")
    for d in report["deviations"]:
        print(f"  - {d}")
    print(f"OVERALL: {'READY' if report['ok'] else 'NOT READY -- see above'}")
    print("=" * 72)
