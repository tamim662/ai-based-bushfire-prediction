import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

from logger_setup import setup_logger
from config import FIRMS_KEY_ENV, FIRMS_CSV, SOURCES, VIC_BBOX, START_DATE, END_DATE, CHUNK_DAYS

logger, _ = setup_logger()

def daterange_chunks(start_date, end_date, chunk_days=5):
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    cur = start
    while cur <= end:
        yield cur, min(cur + timedelta(days=chunk_days - 1), end)
        cur += timedelta(days=chunk_days)

def run_firms_download(force: bool = False) -> str:
    if FIRMS_CSV.exists() and not force:
        logger.info(f"FIRMS cache HIT: using existing file {FIRMS_CSV}")
        return str(FIRMS_CSV)

    key = os.environ.get(FIRMS_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"Missing FIRMS key. Export it: export {FIRMS_KEY_ENV}='YOUR_KEY'")

    all_dfs = []
    for src in SOURCES:
        logger.info(f"=== Downloading source: {src} ===")
        for (chunk_start, chunk_end) in daterange_chunks(START_DATE, END_DATE, chunk_days=CHUNK_DAYS):
            day_range = (chunk_end - chunk_start).days + 1
            date_str = chunk_start.strftime("%Y-%m-%d")

            url = (
                f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                f"{key}/{src}/{VIC_BBOX}/{day_range}/{date_str}"
            )
            logger.info(f"GET {url}")
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200 or len(resp.text.strip()) < 10:
                continue

            df = pd.read_csv(StringIO(resp.text))
            if df.empty:
                continue

            df["source"] = src
            df["chunk_start"] = date_str
            all_dfs.append(df)

    if all_dfs:
        viirs_raw = pd.concat(all_dfs, ignore_index=True)
    else:
        viirs_raw = pd.DataFrame()

    FIRMS_CSV.parent.mkdir(parents=True, exist_ok=True)
    viirs_raw.to_csv(FIRMS_CSV, index=False)
    logger.info(f"Saved FIRMS: {FIRMS_CSV} rows={len(viirs_raw)}")
    return str(FIRMS_CSV)
