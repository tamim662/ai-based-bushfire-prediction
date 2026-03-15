# source/run_pipeline.py
import argparse
from pathlib import Path

from logger_setup import setup_logger
from config import (
    FIRMS_CSV, VIIRS_WITH_EVENT_CSV, EVENTS_VIC_CSV,
    TENSORS_DIR, DATASETS_DIR, ERA5_LOCAL_NC
)

from firms_download import run_firms_download
from events_cluster import run_events
from samples_make import make_candidate_samples
from himawari_patches import build_himawari_quota_with_cleanup
from era5_features import build_era5_sequences
from dataset_build import filter_by_era5_and_save

logger, _ = setup_logger()


STAGES = ["firms", "events", "samples", "himawari", "era5", "final"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pos-target", type=int, default=20)
    p.add_argument("--neg-target", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=30)
    p.add_argument("--max-rounds", type=int, default=80)

    p.add_argument("--force-firms", action="store_true")

    p.add_argument("--era5-mode", choices=["auto", "skip"], default="auto")
    p.add_argument("--require-era5", action="store_true",
                  help="Final dataset only keeps samples with ERA5")

    # NEW
    p.add_argument("--start-from", choices=STAGES, default="firms",
                  help="Start pipeline from a specific stage")
    p.add_argument("--skip-existing", action="store_true",
                  help="If outputs exist, skip that stage")
    return p.parse_args()


def stage_allowed(start_from: str, stage: str) -> bool:
    return STAGES.index(stage) >= STAGES.index(start_from)


def main():
    args = parse_args()

    # --------------------------
    # Stage 1: FIRMS
    # --------------------------
    firms_csv = None
    if stage_allowed(args.start_from, "firms"):
        logger.info("========== 1) FIRMS Download (cached) ==========")
        if args.skip_existing and FIRMS_CSV.exists() and not args.force_firms:
            logger.info(f"SKIP FIRMS: existing {FIRMS_CSV} size={FIRMS_CSV.stat().st_size/1e6:.1f}MB")
            firms_csv = str(FIRMS_CSV)
        else:
            firms_csv = run_firms_download(force=args.force_firms)
    else:
        firms_csv = str(FIRMS_CSV)
        logger.info(f"START-FROM >= events, using FIRMS_CSV={firms_csv}")

    # --------------------------
    # Stage 2: Events
    # --------------------------
    viirs_with_evt, events_csv = None, None
    if stage_allowed(args.start_from, "events"):
        logger.info("========== 2) Event Clustering ==========")
        if args.skip_existing and VIIRS_WITH_EVENT_CSV.exists() and EVENTS_VIC_CSV.exists():
            logger.info(f"SKIP EVENTS: existing {VIIRS_WITH_EVENT_CSV} and {EVENTS_VIC_CSV}")
            viirs_with_evt = str(VIIRS_WITH_EVENT_CSV)
            events_csv = str(EVENTS_VIC_CSV)
        else:
            viirs_with_evt, events_csv = run_events(firms_csv)
    else:
        viirs_with_evt = str(VIIRS_WITH_EVENT_CSV)
        events_csv = str(EVENTS_VIC_CSV)
        logger.info(f"START-FROM >= samples, using viirs_with_evt={viirs_with_evt}, events_csv={events_csv}")

    # --------------------------
    # Stage 3: Samples
    # --------------------------
    samples_csv = None
    if stage_allowed(args.start_from, "samples"):
        logger.info("========== 3) Candidate Sample Generation (oversample pool) ==========")
        # samples_make normally regenerates each time; we can still let it run
        samples_csv = make_candidate_samples(
            events_csv=events_csv,
            viirs_with_event_csv=viirs_with_evt,
            n_fire_events=max(20, args.pos_target),
            fire_samples_per_event=5,
            n_no_fire_anchors=120,
        )
        logger.info(f"Samples CSV: {samples_csv}")
    else:
        logger.info("START-FROM >= himawari, expecting you already have a samples_csv from previous run.")
        # If you want strictness, you can hardcode a path or raise.
        # We'll assume himawari stage can work from existing candidate samples produced earlier.
        raise RuntimeError("You used --start-from himawari/era5/final, but samples_csv is not provided. Run from samples stage or implement fixed samples_csv path.")

    # --------------------------
    # Stage 4: Himawari tensors
    # --------------------------
    X_path = TENSORS_DIR / "X_img.npy"
    y_path = TENSORS_DIR / "y.npy"
    success_csv_path = TENSORS_DIR / "successful_samples.csv"

    if stage_allowed(args.start_from, "himawari"):
        logger.info("========== 4) Himawari Extraction (quota + cleanup) ==========")

        if args.skip_existing and X_path.exists() and y_path.exists() and success_csv_path.exists():
            logger.info(f"SKIP HIMAWARI: existing tensors found: {X_path}, {y_path}, {success_csv_path}")
            X_out, y_out, success_out = str(X_path), str(y_path), str(success_csv_path)
        else:
            X_out, y_out, success_out = build_himawari_quota_with_cleanup(
                samples_csv=samples_csv,
                pos_target=args.pos_target,
                neg_target=args.neg_target,
                chunk_size=args.chunk_size,
                max_rounds=args.max_rounds,
            )
    else:
        X_out, y_out, success_out = str(X_path), str(y_path), str(success_csv_path)
        logger.info(f"START-FROM >= era5, using X={X_out}, y={y_out}, success_csv={success_out}")

    # --------------------------
    # Stage 5: ERA5
    # --------------------------
    era5_npz = DATASETS_DIR / "era5_sequences.npz"
    if stage_allowed(args.start_from, "era5"):
        logger.info("========== 5) ERA5 Feature Sequences (gate) ==========")

        if args.era5_mode == "skip":
            if args.require_era5:
                raise RuntimeError("You used --require-era5 but set --era5-mode skip.")
            logger.warning("ERA5 skipped. Final dataset will be Himawari-only.")
            # build placeholder sequences
            era5_out = build_era5_sequences(success_out)
        else:
            # If require-era5 and NetCDF missing, fail fast with clear message
            if args.require_era5 and not Path(ERA5_LOCAL_NC).exists():
                raise RuntimeError(
                    f"--require-era5 set, but ERA5_LOCAL_NC not found: {ERA5_LOCAL_NC}. "
                    "Put the ERA5 NetCDF there or run without --require-era5."
                )
            if args.skip_existing and era5_npz.exists():
                logger.info(f"SKIP ERA5: existing {era5_npz} size={era5_npz.stat().st_size} bytes")
                era5_out = str(era5_npz)
            else:
                era5_out = build_era5_sequences(success_out)
    else:
        era5_out = str(era5_npz)
        logger.info(f"START-FROM >= final, using era5_npz={era5_out}")

    # --------------------------
    # Final dataset
    # --------------------------
    logger.info("========== 6) Final dataset build ==========")
    final_npz, final_samples = filter_by_era5_and_save(
        X_img_path=X_out,
        y_path=y_out,
        success_csv=success_out,
        era5_npz=era5_out,
        require_era5=args.require_era5 if args.era5_mode != "skip" else False
    )

    logger.info(f"Pipeline done. Final: {final_npz}")
    logger.info(f"Final samples: {final_samples}")


if __name__ == "__main__":
    main()
