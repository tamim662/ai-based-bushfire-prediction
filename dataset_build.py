# source/dataset_build.py
import numpy as np
import pandas as pd
from pathlib import Path

from logger_setup import setup_logger
from config import PROCESSED_DATASETS

logger, _ = setup_logger()


def filter_by_era5_and_save(
    X_img_path: str,
    y_path: str,
    success_csv: str,
    era5_npz: str,
    require_era5: bool = True,
):
    X = np.load(X_img_path)
    y = np.load(y_path)
    meta = pd.read_csv(success_csv)

    z = np.load(era5_npz)
    era5 = z["era5"]
    ok = z["ok"].astype(bool)

    n_total = len(ok)
    n_ok = int(ok.sum())

    logger.info(f"Dataset build input: X={X.shape} y={y.shape} meta_rows={len(meta)} era5={era5.shape} ok={n_ok}/{n_total} require_era5={require_era5}")

    if require_era5:
        keep = ok
    else:
        keep = np.ones_like(ok, dtype=bool)

    n_keep = int(np.sum(keep))
    if n_keep == 0:
        logger.error(
            "Final dataset would be EMPTY after filtering. "
            f"Reason: require_era5={require_era5} but ok={n_ok}/{n_total}. "
            "Fix ERA5 extraction (NetCDF path/time alignment/vars) or run without --require-era5."
        )

    # Safe guarding: if meta length mismatched, log and align
    if len(meta) != X.shape[0] or len(meta) != y.shape[0]:
        logger.warning(f"Meta length mismatch: meta={len(meta)} X={X.shape[0]} y={y.shape[0]}. Aligning by min length.")
        n = min(len(meta), X.shape[0], y.shape[0], len(ok))
        X = X[:n]
        y = y[:n]
        meta = meta.iloc[:n].reset_index(drop=True)
        era5 = era5[:n]
        ok = ok[:n]
        keep = keep[:n]
        n_total = len(ok)
        n_ok = int(ok.sum())
        n_keep = int(np.sum(keep))
        logger.info(f"After align: total={n_total} ok={n_ok} keep={n_keep}")

    X2 = X[keep]
    y2 = y[keep]
    meta2 = meta.loc[keep].reset_index(drop=True)
    era52 = era5[keep]

    out_dir = Path(PROCESSED_DATASETS)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / "final_dataset.npz"
    out_csv = out_dir / "final_samples.csv"

    np.savez_compressed(out_npz, X=X2, y=y2, era5=era52)
    meta2.to_csv(out_csv, index=False)

    if len(meta2) == 0:
        logger.error(f"Final dataset SAVED but EMPTY: {out_npz} (rows=0). Check earlier ERROR logs.")
    else:
        logger.info(f"Final dataset saved: {out_npz} X={X2.shape} y={y2.shape} era5={era52.shape}")
        logger.info(f"Final samples saved: {out_csv} rows={len(meta2)}")

    return str(out_npz), str(out_csv)
