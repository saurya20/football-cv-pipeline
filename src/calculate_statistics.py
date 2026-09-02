"""
Player Movement & Tracking Statistics Calculator.
Computes distance covered, average speed, top speed, tracking duration,
and associates player identity mapping where available.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    PLAYER_POSITIONS_CSV,
    PLAYER_STATISTICS_CSV,
    PLAYERS_CSV
)


def calculate_movement_statistics(
    positions_csv=PLAYER_POSITIONS_CSV,
    players_csv=PLAYERS_CSV,
    output_csv=PLAYER_STATISTICS_CSV,
    fps=59.94
):
    """
    Calculates detailed tracking and movement statistics for each track.
    """
    print("=" * 60)
    print("CALCULATING PLAYER MOVEMENT STATISTICS")
    print("=" * 60)

    if not Path(positions_csv).exists():
        raise FileNotFoundError(f"Positions file not found: {positions_csv}")

    df = pd.read_csv(positions_csv)
    if len(df) == 0:
        print("No position records to compute statistics.")
        return

    # Load player identity mapping if available
    player_roster = {}
    if Path(players_csv).exists():
        p_df = pd.read_csv(players_csv)
        for _, row in p_df.iterrows():
            t_id = row.get("track_id")
            if pd.notna(t_id) and int(t_id) >= 0:
                player_roster[int(t_id)] = {
                    "player_name": row.get("player_name", "Unknown"),
                    "jersey_number": row.get("jersey_number", ""),
                    "team": row.get("team", "Unknown")
                }

    stats = []

    for track_id, group in df.groupby("track_id"):
        g = group.sort_values("frame").copy()
        frames_tracked = len(g)
        first_frame = int(g["frame"].min())
        last_frame = int(g["frame"].max())
        duration_sec = (last_frame - first_frame) / fps if fps > 0 else 0.0

        coord_sys = g["coordinate_system"].iloc[0] if "coordinate_system" in g.columns else "image_normalized"
        primary_view = g["camera_view"].mode()[0] if "camera_view" in g.columns and len(g["camera_view"]) > 0 else "WIDE"

        # Check if pitch coordinates are valid
        valid_pitch_count = g["pitch_x"].notna().sum()
        use_pitch = (coord_sys == "pitch_world") and (valid_pitch_count >= len(g) * 0.5)

        if use_pitch:
            gx = g["pitch_x"].ffill().bfill()
            gy = g["pitch_y"].ffill().bfill()
            dx = gx.diff()
            dy = gy.diff()
            step_dist = np.sqrt(dx**2 + dy**2)
            total_dist = float(step_dist.sum())
            unit = "meters"
        else:
            dx = g["player_x"].diff()
            dy = g["player_y"].diff()
            step_dist = np.sqrt(dx**2 + dy**2)
            total_dist = float(step_dist.sum())
            unit = "pixels"

        # Speeds
        time_diffs = g["frame"].diff() / fps
        time_diffs = time_diffs.replace(0, np.nan)
        instant_speeds = step_dist / time_diffs
        # Filter extreme jumps / teleportation outliers
        if use_pitch:
            instant_speeds = instant_speeds[instant_speeds <= 12.0]  # max 12 m/s (~43 km/h)
        else:
            instant_speeds = instant_speeds[instant_speeds <= 500.0]

        avg_speed = (total_dist / duration_sec) if duration_sec > 0 else 0.0
        max_speed = float(instant_speeds.quantile(0.95)) if len(instant_speeds.dropna()) > 0 else avg_speed

        # Identity lookup
        identity = player_roster.get(track_id, {
            "player_name": f"Track {track_id}",
            "jersey_number": "",
            "team": "Unknown"
        })

        stats.append({
            "track_id": track_id,
            "player_name": identity["player_name"],
            "jersey_number": identity["jersey_number"],
            "team": identity["team"],
            "frames_tracked": frames_tracked,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "duration_seconds": round(duration_sec, 2),
            "distance": round(total_dist, 2),
            "distance_unit": unit,
            "average_speed": round(avg_speed, 2),
            "max_speed": round(max_speed, 2),
            "average_confidence": round(float(g["confidence"].mean()), 3),
            "primary_camera_view": primary_view,
            "coordinate_system": coord_sys
        })

    stats_df = pd.DataFrame(stats).sort_values("distance", ascending=False).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_csv, index=False)

    print()
    print("=" * 60)
    print("STATISTICS SUMMARY")
    print("=" * 60)
    print(f"Total Unique Tracks Analyzed: {len(stats_df):,}")
    print(f"Saved to:                     {output_csv}")
    print()
    print("Top 15 Tracks by Distance:")
    print("-" * 60)
    print(stats_df[[
        "track_id", "player_name", "team", "duration_seconds", "distance", "distance_unit", "average_speed", "max_speed"
    ]].head(15).to_string(index=False))
    print("=" * 60)

    return stats_df


if __name__ == "__main__":
    calculate_movement_statistics()