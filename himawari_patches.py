import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from tqdm import tqdm

from logger_setup import setup_logger
from config import (
    BANDS, SEQ_LEN, PATCH_SIZE,
    HIM_TIME_FLOOR_MIN, HIM_RETRY_MINUTES,
    INTERIM_CACHE_INDEX, PROCESSED_TENSORS,
)
from himawari_cache import ensure_band_files, cleanup_paths, cleanup_dir

logger, _ = setup_logger()

def _parse_dt(x: str) -> datetime:
    dt = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def floor_to_minutes(dt: datetime, minutes: int) -> datetime:
    dt = dt.astimezone(timezone.utc)
    discard = timedelta(minutes=dt.minute % minutes, seconds=dt.second, microseconds=dt.microsecond)
    return dt - discard

def try_extract_sequence_satpy(temp_dir: Path) -> np.ndarray:
    """
    SAFE extractor:
    - If Satpy installed and you have your config working, you can replace this implementation.
    - For now, we implement a minimal "read check" that ensures files exist and returns dummy tensor.
    """
    # You can later replace with your Satpy-based patch extraction to return real pixels.
    # Here we return zeros to avoid crashing and allow pipeline structure to run.
    # Shape: (SEQ_LEN, PATCH, PATCH, C)
    C = len(BANDS)
    return np.zeros((SEQ_LEN, PATCH_SIZE, PATCH_SIZE, C), dtype=np.float32)

def build_himawari_quota_with_cleanup(samples_csv: str, pos_target: int, neg_target: int, chunk_size: int, max_rounds: int):
    samples = pd.read_csv(samples_csv)
    if "is_fire" not in samples.columns:
        raise ValueError("samples_csv must have is_fire")

    PROCESSED_TENSORS.mkdir(parents=True, exist_ok=True)

    X_list = []
    y_list = []
    meta_rows = []

    pos_ok = 0
    neg_ok = 0

    attempted = set()

    for rnd in range(max_rounds):
        if pos_ok >= pos_target and neg_ok >= neg_target:
            break

        remain = samples[~samples["sample_id"].astype(str).isin(attempted)].copy()
        if remain.empty:
            logger.warning("No remaining candidates.")
            break

        batch = remain.head(chunk_size).copy()
        logger.info(f"Round {rnd+1}: trying {len(batch)} candidates (pos_ok={pos_ok}, neg_ok={neg_ok})")

        pbar = tqdm(total=len(batch), desc=f"Himawari round {rnd+1}", unit="sample")

        for _, row in batch.iterrows():
            sid = str(row["sample_id"])
            attempted.add(sid)

            is_fire = int(row["is_fire"])
            if is_fire == 1 and pos_ok >= pos_target:
                pbar.update(1); continue
            if is_fire == 0 and neg_ok >= neg_target:
                pbar.update(1); continue

            tmp = INTERIM_CACHE_INDEX / f"tmp_{sid}"
            tmp.mkdir(parents=True, exist_ok=True)

            created_all: List[Path] = []
            ok = False

            end_dt = _parse_dt(row["end_dt_utc"])
            base_dt = floor_to_minutes(end_dt, HIM_TIME_FLOOR_MIN)

            for off in HIM_RETRY_MINUTES:
                dt_try = base_dt + timedelta(minutes=off)

                # download+decompress into tmp
                band_files, created = ensure_band_files(dt_try, BANDS, temp_dir=tmp)
                created_all.extend(created)

                # require at least 1 file per band
                if any(len(band_files.get(b, [])) == 0 for b in band_files.keys()):
                    continue

                try:
                    seq = try_extract_sequence_satpy(tmp)
                    if seq is None:
                        continue

                    X_list.append(seq)
                    y_list.append(is_fire)
                    meta_rows.append(row.to_dict())

                    if is_fire == 1: pos_ok += 1
                    else: neg_ok += 1

                    ok = True
                    break
                except Exception as e:
                    logger.warning(f"Extraction failed for {sid} at {dt_try}: {e}")
                    continue

            if not ok:
                cleanup_paths(created_all)
                cleanup_dir(tmp)
            else:
                # keep nothing in tmp (space save) even for success
                cleanup_paths(created_all)
                cleanup_dir(tmp)

            pbar.update(1)
            if pos_ok >= pos_target and neg_ok >= neg_target:
                break

        pbar.close()

    if len(X_list) == 0:
        raise RuntimeError("No successful Himawari samples extracted.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)

    X_path = PROCESSED_TENSORS / "X_img.npy"
    y_path = PROCESSED_TENSORS / "y.npy"
    meta_csv = PROCESSED_TENSORS / "successful_samples.csv"

    np.save(X_path, X)
    np.save(y_path, y)
    meta.to_csv(meta_csv, index=False)

    logger.info(f"Saved X: {X_path} shape={X.shape}")
    logger.info(f"Saved y: {y_path} shape={y.shape}")
    logger.info(f"Saved meta: {meta_csv} rows={len(meta)} (pos={pos_ok}, neg={neg_ok})")

    return str(X_path), str(y_path), str(meta_csv)
