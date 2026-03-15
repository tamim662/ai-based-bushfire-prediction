# source/events_cluster.py
import pandas as pd
from datetime import datetime, timezone

from logger_setup import setup_logger
from config import (
    EVENTS_DIR,
    VIC_BBOX,          # can be "w,s,e,n" string OR (w,s,e,n) tuple/list
    GRID_DEG,          # spatial bin size in degrees (coarse)
    EVENT_TIME_HOURS,  # used for time binning (hours)
)

logger, _ = setup_logger()


def parse_bbox(bbox):
    """
    Accepts either:
      - "west,south,east,north"  (string)
      - (west, south, east, north) (tuple/list)
    Returns 4 floats.
    """
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        w, s, e, n = bbox
        return float(w), float(s), float(e), float(n)

    if isinstance(bbox, str):
        parts = [p.strip() for p in bbox.split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError(f"Bad VIC_BBOX string (need 4 numbers): {bbox}")
        w, s, e, n = parts
        return float(w), float(s), float(e), float(n)

    raise ValueError(f"Unsupported VIC_BBOX type: {type(bbox)}")


def _to_dt_utc(row) -> datetime:
    """
    FIRMS VIIRS rows commonly have:
      - acq_date: YYYY-MM-DD
      - acq_time: HHMM (string/int), sometimes missing
    Returns timezone-aware UTC datetime.
    """
    d = str(row.get("acq_date", "")).strip()
    t = str(row.get("acq_time", "0000")).strip().zfill(4)

    if not d:
        # fallback if acq_date missing (shouldn't happen if cleaned properly)
        return datetime.now(timezone.utc)

    hh, mm = t[:2], t[2:]
    iso = f"{d}T{hh}:{mm}:00+00:00"
    return datetime.fromisoformat(iso).astimezone(timezone.utc)


def load_and_clean_firms(csv_path: str) -> pd.DataFrame:
    """
    Load FIRMS CSV and:
      - ensure lat/lon/acq_date present
      - create dt_utc
      - filter to VIC_BBOX (robust bbox parsing)
    """
    df = pd.read_csv(csv_path)

    required = ["latitude", "longitude", "acq_date"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required FIRMS column: {c}")

    df = df.dropna(subset=["latitude", "longitude", "acq_date"]).copy()

    # datetime
    df["dt_utc"] = df.apply(_to_dt_utc, axis=1)

    # bbox filter
    west, south, east, north = parse_bbox(VIC_BBOX)
    df = df[
        (df["longitude"] >= west) & (df["longitude"] <= east) &
        (df["latitude"] >= south) & (df["latitude"] <= north)
    ].copy()

    return df


def run_events(firms_csv: str):
    """
    Create:
      - viirs_clean.csv
      - viirs_with_event.csv
      - events_vic.csv

    Event logic (simple baseline):
      - spatial cell = floor(lat/GRID_DEG), floor(lon/GRID_DEG)
      - time bin = floor to EVENT_TIME_HOURS
      - event_id = category code of (cell_id + time_bin)
    """
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    viirs = load_and_clean_firms(firms_csv)
    clean_path = EVENTS_DIR / "viirs_clean.csv"
    viirs.to_csv(clean_path, index=False)
    logger.info(f"Saved cleaned FIRMS: {clean_path} rows={len(viirs)}")

    # Spatial bin
    viirs["gy"] = (viirs["latitude"] / GRID_DEG).apply(lambda x: int(x // 1))
    viirs["gx"] = (viirs["longitude"] / GRID_DEG).apply(lambda x: int(x // 1))
    viirs["cell_id"] = viirs["gy"].astype(str) + "_" + viirs["gx"].astype(str)

    # Time bin (floor to EVENT_TIME_HOURS)
    # If EVENT_TIME_HOURS = 24 -> daily bins
    viirs["time_bin"] = viirs["dt_utc"].dt.floor(f"{int(EVENT_TIME_HOURS)}H")

    # Event key -> event_id
    viirs["event_key"] = viirs["cell_id"].astype(str) + "_" + viirs["time_bin"].astype(str)
    viirs["event_id"] = viirs["event_key"].astype("category").cat.codes.astype(int)

    with_event_path = EVENTS_DIR / "viirs_with_event.csv"
    viirs.to_csv(with_event_path, index=False)
    logger.info(f"Saved viirs_with_event: {with_event_path}")

    # Event table
    events = (
        viirs.groupby("event_id")
        .agg(
            n_points=("event_id", "size"),
            start_dt=("dt_utc", "min"),
            end_dt=("dt_utc", "max"),
            mean_lat=("latitude", "mean"),
            mean_lon=("longitude", "mean"),
        )
        .reset_index()
        .sort_values("n_points", ascending=False)
    )

    events_path = EVENTS_DIR / "events_vic.csv"
    events.to_csv(events_path, index=False)
    logger.info(f"Saved events_vic: {events_path} events={len(events)}")

    return str(with_event_path), str(events_path)
