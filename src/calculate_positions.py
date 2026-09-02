"""
Player Position Calculation Module.
Computes ground contact reference points (bottom-center of bounding box),
normalized image coordinates, and maps to pitch-world coordinates (meters)
when calibrated homography is available.
"""
import pandas as pd
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    CLEAN_TRACKING_CSV,
    CAMERA_SEGMENTS_CSV,
    PLAYER_POSITIONS_CSV,
    PITCH_LENGTH_METRES,
    PITCH_WIDTH_METRES
)


def compute_player_positions(
    input_csv=CLEAN_TRACKING_CSV,
    camera_csv=CAMERA_SEGMENTS_CSV,
    output_csv=PLAYER_POSITIONS_CSV,
    frame_width=1920,
    frame_height=1080,
    wide_homography=None
):
    """
    Computes bottom-center foot positions, normalized coordinates,
    and world pitch coordinates.
    """
    print("=" * 60)
    print("CALCULATING PLAYER POSITIONS & COORDINATES")
    print("=" * 60)

    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Clean tracking file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if len(df) == 0:
        print("No tracking records to process.")
        return

    # Bottom-center of bounding box = player ground contact point
    df["player_x"] = ((df["x1"] + df["x2"]) / 2).round(2)
    df["player_y"] = df["y2"].round(2)

    # Normalized image coordinates [0.0, 1.0]
    df["norm_x"] = (df["player_x"] / frame_width).clip(0.0, 1.0).round(4)
    df["norm_y"] = (df["player_y"] / frame_height).clip(0.0, 1.0).round(4)

    # Merge camera view if available
    df["camera_view"] = "WIDE"
    if Path(camera_csv).exists():
        cam_df = pd.read_csv(camera_csv)
        if len(cam_df) > 0:
            for _, c_row in cam_df.iterrows():
                s_f = int(c_row["start_frame"])
                e_f = int(c_row["end_frame"])
                view = c_row["camera_view"]
                df.loc[(df["frame"] >= s_f) & (df["frame"] <= e_f), "camera_view"] = view

    # Default coordinate system: image_normalized
    df["pitch_x"] = np.nan
    df["pitch_y"] = np.nan
    df["coordinate_system"] = "image_normalized"

    # If wide homography matrix is not passed, load from PitchCalibrator
    if wide_homography is None:
        try:
            from src.calibration import PitchCalibrator
            calibrator = PitchCalibrator()
            wide_homography = calibrator.h_wide
        except Exception:
            wide_homography = None

    # If wide homography matrix is provided / calibrated, calculate pitch coordinates
    if wide_homography is not None:
        try:
            wide_mask = df["camera_view"] == "WIDE"
            if wide_mask.sum() > 0:
                pts = df.loc[wide_mask, ["player_x", "player_y"]].values.reshape(-1, 1, 2).astype(np.float32)
                transformed = cv2.perspectiveTransform(pts, wide_homography)
                df.loc[wide_mask, "pitch_x"] = np.clip(transformed[:, 0, 0], 0, PITCH_LENGTH_METRES).round(2)
                df.loc[wide_mask, "pitch_y"] = np.clip(transformed[:, 0, 1], 0, PITCH_WIDTH_METRES).round(2)
                df.loc[wide_mask, "coordinate_system"] = "pitch_world"
        except Exception as e:
            print(f"Homography transform warning: {e}. Falling back to image_normalized coordinates.")

    positions = df[[
        "frame",
        "timestamp",
        "track_id",
        "camera_view",
        "player_x",
        "player_y",
        "norm_x",
        "norm_y",
        "pitch_x",
        "pitch_y",
        "coordinate_system",
        "confidence"
    ]].sort_values(["track_id", "frame"]).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(output_csv, index=False)

    print()
    print("=" * 60)
    print("POSITION CALCULATION COMPLETE")
    print("=" * 60)
    print(f"Total Position Records:       {len(positions):,}")
    print(f"Unique Tracks:                {positions['track_id'].nunique():,}")
    print(f"Coordinate Systems Used:      {positions['coordinate_system'].value_counts().to_dict()}")
    print(f"Saved to:                     {output_csv}")
    print("=" * 60)

    return positions


if __name__ == "__main__":
    compute_player_positions()