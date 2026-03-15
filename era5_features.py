# source/era5_features.py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

from logger_setup import setup_logger
from config import ERA5_LOOKBACK_HOURS, ERA5_VARS, ERA5_LOCAL_NC, PROCESSED_DATASETS

logger, _ = setup_logger()


def _parse_dt(x) -> datetime:
    dt = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def era5_available_local() -> bool:
    return Path(ERA5_LOCAL_NC).exists()


def _open_era5_dataset():
    """
    Open ERA5 NetCDF using xarray. We keep it lazy (not loading all into RAM).
    """
    try:
        import xarray as xr
    except Exception as e:
        raise RuntimeError(
            "xarray is required to read ERA5 NetCDF. Install with:\n"
            "  pip install xarray netCDF4\n"
            f"Original error: {e}"
        )
    ds = xr.open_dataset(ERA5_LOCAL_NC)
    return ds


def _find_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords:
            return c
        if c in ds.variables:
            return c
    return None


def _select_point_and_time(ds, lat, lon, times_utc):
    """
    Select nearest grid point and nearest time stamps.
    We auto-detect coord names commonly used in ERA5 files.
    """
    import xarray as xr  # safe here because open already required it

    lat_name = _find_coord_name(ds, ["latitude", "lat"])
    lon_name = _find_coord_name(ds, ["longitude", "lon"])
    time_name = _find_coord_name(ds, ["time", "valid_time"])

    if not lat_name or not lon_name or not time_name:
        raise RuntimeError(
            f"ERA5 NetCDF missing expected coords. Found coords={list(ds.coords)} vars={list(ds.data_vars)}"
        )

    # Ensure lon in same domain as file (0..360 vs -180..180)
    lon_val = float(lon)
    ds_lon = ds[lon_name]
    try:
        lon_min = float(ds_lon.min())
        lon_max = float(ds_lon.max())
    except Exception:
        lon_min, lon_max = -180.0, 180.0

    if lon_min >= 0 and lon_val < 0:
        lon_val = lon_val % 360.0

    # Build xarray time index
    # times_utc is list of python datetime with tzinfo UTC
    # Convert to numpy datetime64
    times64 = np.array([np.datetime64(t.replace(tzinfo=None)) for t in times_utc])

    # nearest selection for each time step
    # We select the point first then time to minimize work.
    point = ds.sel(
        {lat_name: float(lat), lon_name: lon_val},
        method="nearest"
    )

    # now align time
    point = point.sel({time_name: times64}, method="nearest")
    return point, time_name


def build_era5_sequence_for_sample_local(sample_row: pd.Series, ds=None) -> np.ndarray:
    """
    Returns array shape: (ERA5_LOOKBACK_HOURS, len(ERA5_VARS))
    Variables must exist in ds (data_vars). If any var missing -> NaNs for that var.
    """
    # must have these columns in success_csv
    lat = float(sample_row["lat"])
    lon = float(sample_row["lon"])
    dt = _parse_dt(sample_row["dt_utc"])

    # We want last N hours ending at dt floored to hour
    dt_hour = dt.replace(minute=0, second=0, microsecond=0)
    times = [dt_hour - timedelta(hours=h) for h in range(ERA5_LOOKBACK_HOURS - 1, -1, -1)]

    if ds is None:
        ds = _open_era5_dataset()

    try:
        point, _ = _select_point_and_time(ds, lat, lon, times)
    except Exception as e:
        logger.warning(f"ERA5 select failed for sample_id={sample_row.get('sample_id','?')} dt={dt_hour.isoformat()} lat={lat:.4f} lon={lon:.4f} err={e}")
        return np.full((ERA5_LOOKBACK_HOURS, len(ERA5_VARS)), np.nan, dtype=np.float32)

    feats = np.full((ERA5_LOOKBACK_HOURS, len(ERA5_VARS)), np.nan, dtype=np.float32)

    for j, v in enumerate(ERA5_VARS):
        if v not in point.data_vars:
            logger.warning(f"ERA5 var missing in NetCDF: '{v}'. Available vars: {list(point.data_vars)}")
            continue

        try:
            arr = point[v].values
            # expected shape (T,) or (T,1,1) etc. squeeze to (T,)
            arr = np.asarray(arr).squeeze()
            if arr.shape[0] != ERA5_LOOKBACK_HOURS:
                # If time alignment returns different length, fallback to NaNs
                logger.warning(f"ERA5 var '{v}' returned shape {arr.shape}, expected ({ERA5_LOOKBACK_HOURS},).")
                continue
            feats[:, j] = arr.astype(np.float32)
        except Exception as e:
            logger.warning(f"ERA5 extraction failed var='{v}' err={e}")

    return feats


def build_era5_sequences(success_csv: str) -> str:
    """
    Build per-sample ERA5 sequences and save as one .npz:
      - era5: (N, T, F)
      - ok:   (N,) True if at least one non-NaN exists (or stricter rule if you want)
    """
    df = pd.read_csv(success_csv)
    out_path = Path(PROCESSED_DATASETS) / "era5_sequences.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"ERA5_LOCAL_NC path: {ERA5_LOCAL_NC}")
    logger.info(f"ERA5_LOCAL_NC exists: {Path(ERA5_LOCAL_NC).exists()}")
    logger.info(f"Building ERA5 sequences for N={len(df)} samples, lookback={ERA5_LOOKBACK_HOURS}h vars={ERA5_VARS}")

    if len(df) == 0:
        seqs = np.empty((0, ERA5_LOOKBACK_HOURS, len(ERA5_VARS)), dtype=np.float32)
        ok_mask = np.array([], dtype=bool)
        np.savez_compressed(out_path, era5=seqs, ok=ok_mask)
        logger.warning(f"Saved ERA5 sequences (EMPTY input): {out_path} shape={seqs.shape} ok=0/0")
        return str(out_path)

    if not era5_available_local():
        # No NetCDF -> everything missing
        seqs = np.full((len(df), ERA5_LOOKBACK_HOURS, len(ERA5_VARS)), np.nan, dtype=np.float32)
        ok_mask = np.zeros((len(df),), dtype=bool)
        np.savez_compressed(out_path, era5=seqs, ok=ok_mask)
        logger.error(
            f"ERA5 NetCDF not found. Saved placeholder NaNs: {out_path} shape={seqs.shape} ok=0/{len(df)}. "
            f"Provide ERA5 netcdf at: {ERA5_LOCAL_NC}"
        )
        return str(out_path)

    # open once
    ds = _open_era5_dataset()

    seqs = []
    ok_mask = []

    for i, row in df.iterrows():
        seq = build_era5_sequence_for_sample_local(row, ds=ds)

        # Gate rule: strict — require NO NaNs at all for "ok"
        # (You can relax this later if you want.)
        ok = not np.any(np.isnan(seq))

        ok_mask.append(ok)
        seqs.append(seq)

        if (i + 1) % 5 == 0 or (i + 1) == len(df):
            logger.info(f"ERA5 progress: {i+1}/{len(df)} ok_so_far={int(np.sum(ok_mask))}")

    seqs = np.stack(seqs, axis=0)
    ok_mask = np.array(ok_mask, dtype=bool)

    np.savez_compressed(out_path, era5=seqs, ok=ok_mask)
    logger.info(f"Saved ERA5 sequences: {out_path} shape={seqs.shape} ok={ok_mask.sum()}/{len(ok_mask)}")
    return str(out_path)
