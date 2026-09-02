"""
Track Cleaning Module.
Cleans raw tracking data by removing invalid track IDs, non-gameplay frame detections,
low-confidence detections, and short-lived spurious tracks.
"""
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRACKING_CSV,
    SCOREBOARD_SEGMENTS_CSV,
    CLEAN_TRACKING_CSV,
    MIN_TRACK_FRAMES,
    MIN_CONFIDENCE
)


def clean_tracking_data(
    tracking_csv=TRACKING_CSV,
    segments_csv=SCOREBOARD_SEGMENTS_CSV,
    output_csv=CLEAN_TRACKING_CSV,
    min_confidence=MIN_CONFIDENCE,
    min_track_frames=MIN_TRACK_FRAMES
):
    """
    Cleans raw tracking detections and exports clean_tracking.csv.
    """
    print("=" * 60)
    print("TRACK CLEANING & FILTERING")
    print("=" * 60)

    if not Path(tracking_csv).exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_csv}")

    tracking = pd.read_csv(tracking_csv)
    orig_records = len(tracking)
    orig_tracks = tracking["track_id"].nunique()

    print(f"1. Original raw records:          {orig_records:,}")
    print(f"   Original unique tracks:        {orig_tracks:,}")

    # 1. Remove invalid IDs
    tracking = tracking[tracking["track_id"] >= 0].copy()
    valid_id_records = len(tracking)
    print(f"2. After removing invalid IDs:    {valid_id_records:,} records")

    # 2. Filter strictly by scoreboard segments
    if Path(segments_csv).exists():
        segments = pd.read_csv(segments_csv)
        if len(segments) > 0:
            # Build interval check
            valid_frames = []
            for _, seg in segments.iterrows():
                valid_frames.append((int(seg["start_frame"]), int(seg["end_frame"])))

            def is_in_gameplay(f):
                return any(start <= f <= end for start, end in valid_frames)

            tracking = tracking[tracking["frame"].apply(is_in_gameplay)].copy()
            print(f"3. After scoreboard filtering:    {len(tracking):,} records")

    # 3. Confidence filtering
    tracking = tracking[tracking["confidence"] >= min_confidence].copy()
    print(f"4. After confidence filtering:    {len(tracking):,} records")

    # 4. Remove short-lived tracks
    track_lengths = tracking.groupby("track_id")["frame"].count()
    persistent_tracks = track_lengths[track_lengths >= min_track_frames].index
    tracking = tracking[tracking["track_id"].isin(persistent_tracks)].copy()

    # Sort
    tracking = tracking.sort_values(["frame", "track_id"]).reset_index(drop=True)

    # Save
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    tracking.to_csv(output_csv, index=False)

    final_records = len(tracking)
    final_tracks = tracking["track_id"].nunique()

    print()
    print("=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Final records:                    {final_records:,}")
    print(f"Final unique tracks:              {final_tracks:,}")
    print(f"Output saved to:                  {output_csv}")
    print("=" * 60)

    return tracking


if __name__ == "__main__":
    clean_tracking_data()