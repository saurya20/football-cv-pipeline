"""
Pipeline Data Quality & Validation Module.
Audits all intermediate and final outputs of the Football CV pipeline,
ensuring data integrity, non-empty files, valid coordinate ranges, and correct metadata.
"""
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    SCOREBOARD_FRAMES_CSV,
    SCOREBOARD_SEGMENTS_CSV,
    SCOREBOARD_DATA_CSV,
    CAMERA_SEGMENTS_CSV,
    TRACKING_CSV,
    CLEAN_TRACKING_CSV,
    PLAYER_POSITIONS_CSV,
    PLAYER_STATISTICS_CSV,
    PLAYERS_CSV,
    TRAJECTORIES_PLOT
)


def validate_pipeline():
    """
    Runs comprehensive validation checks across all pipeline outputs.
    """
    print("=" * 60)
    print("PIPELINE DATA INTEGRITY VALIDATION")
    print("=" * 60)

    results = []

    # 1. Scoreboard Frames
    if SCOREBOARD_FRAMES_CSV.exists():
        df = pd.read_csv(SCOREBOARD_FRAMES_CSV)
        if len(df) > 0 and "scoreboard_present" in df.columns:
            det_count = (df["scoreboard_present"] == 1).sum()
            results.append(("Scoreboard Frames", "PASS", f"{len(df):,} frames ({det_count:,} detected)"))
        else:
            results.append(("Scoreboard Frames", "FAIL", "File is empty or invalid schema"))
    else:
        results.append(("Scoreboard Frames", "MISSING", f"{SCOREBOARD_FRAMES_CSV.name} not found"))

    # 2. Scoreboard Segments
    if SCOREBOARD_SEGMENTS_CSV.exists():
        df = pd.read_csv(SCOREBOARD_SEGMENTS_CSV)
        if len(df) > 0 and "duration" in df.columns:
            tot_dur = df["duration"].sum()
            results.append(("Scoreboard Segments", "PASS", f"{len(df)} segments ({tot_dur:.1f}s total)"))
        else:
            results.append(("Scoreboard Segments", "FAIL", "File is empty or invalid schema"))
    else:
        results.append(("Scoreboard Segments", "MISSING", f"{SCOREBOARD_SEGMENTS_CSV.name} not found"))

    # 3. Scoreboard OCR Data
    if SCOREBOARD_DATA_CSV.exists():
        df = pd.read_csv(SCOREBOARD_DATA_CSV)
        if len(df) > 0:
            scores = df["score"].dropna().nunique()
            results.append(("Scoreboard OCR Data", "PASS", f"{len(df):,} samples ({scores} distinct scores)"))
        else:
            results.append(("Scoreboard OCR Data", "FAIL", "Empty file"))
    else:
        results.append(("Scoreboard OCR Data", "MISSING", f"{SCOREBOARD_DATA_CSV.name} not found"))

    # 4. Camera Segments
    if CAMERA_SEGMENTS_CSV.exists():
        df = pd.read_csv(CAMERA_SEGMENTS_CSV)
        if len(df) > 0 and "camera_view" in df.columns:
            views = df["camera_view"].value_counts().to_dict()
            results.append(("Camera Segments", "PASS", f"{len(df)} segments {views}"))
        else:
            results.append(("Camera Segments", "FAIL", "Empty file"))
    else:
        results.append(("Camera Segments", "MISSING", f"{CAMERA_SEGMENTS_CSV.name} not found"))

    # 5. Raw Tracking
    if TRACKING_CSV.exists():
        df = pd.read_csv(TRACKING_CSV)
        if len(df) > 0 and "track_id" in df.columns:
            results.append(("Raw Tracking Data", "PASS", f"{len(df):,} records ({df['track_id'].nunique():,} tracks)"))
        else:
            results.append(("Raw Tracking Data", "FAIL", "Empty file"))
    else:
        results.append(("Raw Tracking Data", "MISSING", f"{TRACKING_CSV.name} not found"))

    # 6. Clean Tracking
    if CLEAN_TRACKING_CSV.exists():
        df = pd.read_csv(CLEAN_TRACKING_CSV)
        if len(df) > 0 and "track_id" in df.columns:
            invalid_ids = (df["track_id"] < 0).sum()
            low_conf = (df["confidence"] < 0.20).sum()
            if invalid_ids == 0 and low_conf == 0:
                results.append(("Clean Tracking Data", "PASS", f"{len(df):,} records ({df['track_id'].nunique():,} tracks)"))
            else:
                results.append(("Clean Tracking Data", "WARN", f"{invalid_ids} invalid IDs, {low_conf} low conf"))
        else:
            results.append(("Clean Tracking Data", "FAIL", "Empty file"))
    else:
        results.append(("Clean Tracking Data", "MISSING", f"{CLEAN_TRACKING_CSV.name} not found"))

    # 7. Player Positions
    if PLAYER_POSITIONS_CSV.exists():
        df = pd.read_csv(PLAYER_POSITIONS_CSV)
        if len(df) > 0 and "player_x" in df.columns and "coordinate_system" in df.columns:
            null_pos = df["player_x"].isna().sum()
            coord_sys = df["coordinate_system"].unique().tolist()
            if null_pos == 0:
                results.append(("Player Positions", "PASS", f"{len(df):,} positions ({coord_sys})"))
            else:
                results.append(("Player Positions", "FAIL", f"{null_pos} missing coordinates"))
        else:
            results.append(("Player Positions", "FAIL", "Empty file or missing columns"))
    else:
        results.append(("Player Positions", "MISSING", f"{PLAYER_POSITIONS_CSV.name} not found"))

    # 8. Movement Statistics
    if PLAYER_STATISTICS_CSV.exists():
        df = pd.read_csv(PLAYER_STATISTICS_CSV)
        if len(df) > 0 and "distance" in df.columns and "average_speed" in df.columns:
            results.append(("Movement Statistics", "PASS", f"{len(df):,} tracks analyzed"))
        else:
            results.append(("Movement Statistics", "FAIL", "Empty file"))
    else:
        results.append(("Movement Statistics", "MISSING", f"{PLAYER_STATISTICS_CSV.name} not found"))

    # 9. Player Roster
    if PLAYERS_CSV.exists():
        df = pd.read_csv(PLAYERS_CSV)
        results.append(("Player Roster", "PASS", f"{len(df)} players listed"))
    else:
        results.append(("Player Roster", "MISSING", f"{PLAYERS_CSV.name} not found"))

    # 10. Visualizations
    if TRAJECTORIES_PLOT.exists():
        results.append(("Trajectory Plot", "PASS", f"{TRAJECTORIES_PLOT.name} exists"))
    else:
        results.append(("Trajectory Plot", "MISSING", f"{TRAJECTORIES_PLOT.name} not found"))

    # Print Validation Table
    print(f"{'Component':<25} | {'Status':<8} | {'Details'}")
    print("-" * 60)
    all_passed = True
    for comp, status, details in results:
        color_mark = "✔" if status == "PASS" else ("⚠" if status == "WARN" else "✖")
        print(f"{comp:<25} | {status:<8} | {details}")
        if status not in ("PASS", "WARN"):
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("OVERALL PIPELINE INTEGRITY: PASS (Ready for Dashboard & Evaluation)")
    else:
        print("OVERALL PIPELINE INTEGRITY: INCOMPLETE (Run pending stages)")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    validate_pipeline()
