# source/config.py
from __future__ import annotations
from pathlib import Path

# =============================
# Project root + directory tree
# =============================
# Structure:
# BUSHFIRE_DETECTION/
#   artifacts/
#   data/
#   reports/
#   source/   <-- this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw
FIRMS_DIR = RAW_DIR / "firms"
HIMAWARI_DIR = RAW_DIR / "himawari"
ERA5_DIR = RAW_DIR / "era5"

# Interim
EVENTS_DIR = INTERIM_DIR / "events"
SAMPLES_DIR = INTERIM_DIR / "samples"
CACHE_INDEX_DIR = INTERIM_DIR / "cache_index"

# Processed
TENSORS_DIR = PROCESSED_DIR / "tensors"
DATASETS_DIR = PROCESSED_DIR / "datasets"

# Artifacts subdirs
MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LOGS_DIR = ARTIFACTS_DIR / "logs"

# Create folders
for d in [
    ARTIFACTS_DIR, DATA_DIR, REPORTS_DIR,
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
    FIRMS_DIR, HIMAWARI_DIR, ERA5_DIR,
    EVENTS_DIR, SAMPLES_DIR, CACHE_INDEX_DIR,
    TENSORS_DIR, DATASETS_DIR,
    MODELS_DIR, SCALERS_DIR, FIGURES_DIR, LOGS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# =============================
# FIRMS settings
# =============================
FIRMS_KEY_ENV = "FIRMS_MAP_KEY"

# Victoria bbox: west,south,east,north (lon,lat)
VIC_BBOX = "140.95,-39.30,150.10,-33.90"

SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
]

START_DATE = "2018-10-01"
END_DATE = "2024-03-31"
CHUNK_DAYS = 5

FIRMS_CSV = FIRMS_DIR / f"viirs_victoria_{START_DATE}_to_{END_DATE}.csv"

# =============================
# Event clustering parameters
# =============================
# old code compatibility: GRID_DEG often used for spatial binning
GRID_DEG = 0.045  # ~5km

# new code parameters (if used)
EVENT_GRID_KM = 5.0
EVENT_TIME_HOURS = 24

# Outputs
VIIRS_CLEAN_CSV = EVENTS_DIR / "viirs_clean.csv"
VIIRS_WITH_EVENT_CSV = EVENTS_DIR / "viirs_with_event.csv"
EVENTS_VIC_CSV = EVENTS_DIR / "events_vic.csv"

# =============================
# Himawari (NOAA S3 + Satpy)
# =============================
SEQ_LEN = 6
PATCH_SIZE = 32

# Bands (3 channels example)
BANDS = ["B03", "B07", "B13"]

HIM_TIME_FLOOR_MIN = 10
HIM_RETRY_MINUTES = [0, -10, 10, -20, 20]

NOAA_BUCKET = "noaa-himawari8"
NOAA_PRODUCT = "AHI-L1b-FLDK"

# =============================
# ERA5 (gate)
# =============================
ERA5_LOOKBACK_HOURS = 24
ERA5_VARS = ["t2m", "u10", "v10", "tp"]
ERA5_LOCAL_NC = ERA5_DIR / "era5_victoria.nc"

# ==========================================================
# Backward-compatible aliases (IMPORTANT for your old modules)
# ==========================================================
# Older modules might import these exact names:
RAW_FIRMS = FIRMS_DIR
RAW_HIM = HIMAWARI_DIR
RAW_ERA5 = ERA5_DIR

INTERIM_EVENTS = EVENTS_DIR
INTERIM_SAMPLES = SAMPLES_DIR
INTERIM_CACHE_INDEX = CACHE_INDEX_DIR

PROCESSED_TENSORS = TENSORS_DIR
PROCESSED_DATASETS = DATASETS_DIR

# Some old code expects shorter or alternate names:
EVENTS_PATH = EVENTS_DIR
SAMPLES_PATH = SAMPLES_DIR
CACHE_INDEX_PATH = CACHE_INDEX_DIR
TENSORS_PATH = TENSORS_DIR
DATASETS_PATH = DATASETS_DIR
