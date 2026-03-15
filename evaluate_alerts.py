import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score

from config import DATASETS_DIR, MODELS_DIR, EVENTS_DIR
from logger_setup import setup_logger

def apply_persistence(df, prob_col="p", thr=0.6, k=2):
    df = df.sort_values(["cell_id", "end_time_utc"]).copy()
    df["alert_raw"] = (df[prob_col] >= thr).astype(int)

    def pers(g):
        a = g["alert_raw"].to_numpy()
        out = np.zeros_like(a)
        run = 0
        for i in range(len(a)):
            run = run + 1 if a[i] == 1 else 0
            out[i] = 1 if run >= k else 0
        g["alert"] = out
        return g

    return df.groupby("cell_id", group_keys=False).apply(pers)

def false_alarms_per_day(alert_df):
    alert_df["day"] = pd.to_datetime(alert_df["end_time_utc"], utc=True).dt.date
    days_with_alert = alert_df.loc[alert_df["alert"] == 1, "day"].nunique()
    total_days = alert_df["day"].nunique()
    return days_with_alert / max(total_days, 1)

def lead_time_per_event(alert_df, events_df):
    # v1: match by cell_id
    out = []
    for _, ev in events_df.iterrows():
        cid = ev["cell_id"]
        t0 = pd.to_datetime(ev["t0_time"], utc=True)

        dfc = alert_df[(alert_df["cell_id"] == cid) & (alert_df["alert"] == 1)].copy()
        dfc = dfc[dfc["end_time_utc"] <= t0].sort_values("end_time_utc")

        if len(dfc) == 0:
            out.append(np.nan)
        else:
            first_alert = pd.to_datetime(dfc["end_time_utc"].iloc[0], utc=True)
            out.append((t0 - first_alert).total_seconds() / 60.0)

    return pd.Series(out, name="lead_time_min")

def evaluate(thr=0.6, k=2):
    logger, _ = setup_logger()

    X_img = np.load(DATASETS_DIR / "X_img_test.npy")
    X_met = np.load(DATASETS_DIR / "X_met_test.npy")
    y = np.load(DATASETS_DIR / "y_test.npy")
    meta = pd.read_csv(DATASETS_DIR / "meta_test.csv")

    meta["end_time_utc"] = pd.to_datetime(meta["end_time_utc"], utc=True)

    model = tf.keras.models.load_model(MODELS_DIR / "cnn_lstm_himawari_era5.keras")
    p = model.predict({"img_seq": X_img, "met_seq": X_met}, batch_size=128).ravel()

    ap = average_precision_score(y, p)
    logger.info(f"Test PR-AUC: {ap:.4f}")

    meta["p"] = p
    alert_df = apply_persistence(meta, prob_col="p", thr=thr, k=k)

    fa = false_alarms_per_day(alert_df)
    logger.info(f"False alarms/day: {fa:.4f} (thr={thr}, k={k})")

    events = pd.read_csv(EVENTS_DIR / "events_vic.csv")
    events["t0_time"] = pd.to_datetime(events["t0_time"], utc=True)

    lt = lead_time_per_event(alert_df, events)
    logger.info(f"Lead time (min) median={np.nanmedian(lt):.1f} mean={np.nanmean(lt):.1f} (NaNs=missed events)")

    return alert_df, lt
