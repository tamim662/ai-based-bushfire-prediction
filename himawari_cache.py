import bz2
import boto3
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union, Optional

from botocore import UNSIGNED
from botocore.client import Config as BotoConfig

from logger_setup import setup_logger
from config import NOAA_BUCKET, NOAA_PRODUCT

logger, _ = setup_logger()

def _s3_client_unsigned():
    return boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))

def _normalize_band(band: Union[int, str]) -> str:
    if isinstance(band, int):
        return f"B{band:02d}"
    b = str(band).strip().upper()
    if not b.startswith("B"):
        b = "B" + b
    if len(b) == 2:
        b = "B0" + b[1:]
    return b

def _prefix_for_time(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    return f"{NOAA_PRODUCT}/{dt:%Y/%m/%d/%H%M}/"

def list_files_for_time(dt: datetime) -> List[str]:
    s3 = _s3_client_unsigned()
    prefix = _prefix_for_time(dt)
    keys = []
    token = None
    while True:
        kwargs = {"Bucket": NOAA_BUCKET, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys

def ensure_band_files(
    dt: datetime,
    bands: List[Union[str, int]],
    temp_dir: Path,
) -> Tuple[Dict[str, List[Path]], List[Path]]:
    """
    Downloads into temp_dir so caller can delete everything if sample fails.
    Returns band_files and created DAT files list.
    """
    dt = dt.astimezone(timezone.utc)
    s3 = _s3_client_unsigned()

    keys = list_files_for_time(dt)
    if not keys:
        logger.warning(f"No S3 objects found for {dt:%Y-%m-%d %H:%M}Z")
        return ({_normalize_band(b): [] for b in bands}, [])

    temp_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, List[Path]] = {}
    created: List[Path] = []

    for b in bands:
        band_name = _normalize_band(b)
        token = f"_{band_name}_"
        band_keys = [k for k in keys if token in Path(k).name]

        local_paths: List[Path] = []
        for k in band_keys:
            fname = Path(k).name
            bz2_path = temp_dir / fname
            dat_path = temp_dir / fname.replace(".bz2", "")

            if dat_path.exists():
                local_paths.append(dat_path)
                continue

            if not bz2_path.exists():
                try:
                    logger.info(f"Cache MISS: downloading {fname}")
                    s3.download_file(NOAA_BUCKET, k, str(bz2_path))
                except Exception as e:
                    logger.warning(f"Download failed {k}: {e}")
                    continue

            try:
                logger.info(f"Decompressing: {bz2_path.name} -> {dat_path.name}")
                with bz2.open(bz2_path, "rb") as f_in, open(dat_path, "wb") as f_out:
                    f_out.write(f_in.read())
                created.append(dat_path)

                try:
                    bz2_path.unlink()
                    logger.info(f"Deleted compressed file: {bz2_path.name}")
                except Exception:
                    pass

                local_paths.append(dat_path)
            except Exception as e:
                logger.warning(f"Decompress failed {bz2_path}: {e}")
                try:
                    if bz2_path.exists():
                        bz2_path.unlink()
                except Exception:
                    pass

        out[band_name] = local_paths

    return out, created

def cleanup_paths(paths: List[Path]):
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except Exception as e:
            logger.warning(f"Cleanup failed for {p}: {e}")

def cleanup_dir(d: Path):
    try:
        if d.exists():
            for fp in d.glob("*"):
                try: fp.unlink()
                except Exception: pass
            d.rmdir()
    except Exception:
        pass
