import pandas as pd
import uuid
from logger_setup import setup_logger
from config import INTERIM_SAMPLES

logger, _ = setup_logger()

def make_candidate_samples(
    events_csv: str,
    viirs_with_event_csv: str,
    n_fire_events: int = 20,
    fire_samples_per_event: int = 5,
    n_no_fire_anchors: int = 60,
) -> str:
    INTERIM_SAMPLES.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(events_csv)
    viirs = pd.read_csv(viirs_with_event_csv)

    # Choose top events (more points = stronger fire event)
    fire_event_ids = (
        events.sort_values("n_points", ascending=False)
        .head(n_fire_events)["event_id"]
        .astype(int)
        .tolist()
    )

    rows = []

    # FIRE candidates
    for eid in fire_event_ids:
        pts = viirs[viirs["event_id"] == eid].copy()
        if pts.empty:
            continue
        pts = pts.sample(min(len(pts), fire_samples_per_event), random_state=42)

        for _, r in pts.iterrows():
            rows.append({
                "sample_id": str(uuid.uuid4()),
                "is_fire": 1,
                "event_id": int(eid),
                "lat": float(r["latitude"]),
                "lon": float(r["longitude"]),
                "end_dt_utc": str(r["dt_utc"]),
            })

    # NO-FIRE anchors (simple: random points; later improve with far-from-fire)
    nf = viirs.sample(min(len(viirs), n_no_fire_anchors), random_state=123)
    for _, r in nf.iterrows():
        rows.append({
            "sample_id": str(uuid.uuid4()),
            "is_fire": 0,
            "event_id": -1,
            "lat": float(r["latitude"]),
            "lon": float(r["longitude"]),
            "end_dt_utc": str(r["dt_utc"]),
        })

    out = pd.DataFrame(rows)
    out_path = INTERIM_SAMPLES / "samples_candidates.csv"
    out.to_csv(out_path, index=False)

    logger.info(f"Saved candidate samples: {out_path} rows={len(out)} (pos={sum(out.is_fire==1)}, neg={sum(out.is_fire==0)})")
    return str(out_path)
