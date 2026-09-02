"""
Trajectory & Analytics Visualization Module.
Plots player movement paths, trajectories, and gameplay coverage timelines.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    PLAYER_POSITIONS_CSV,
    SCOREBOARD_SEGMENTS_CSV,
    SCOREBOARD_FRAMES_CSV,
    TRAJECTORIES_PLOT,
    CAMERA_COVERAGE_PLOT,
    PITCH_LENGTH_METRES,
    PITCH_WIDTH_METRES
)


def draw_futsal_pitch(ax, length=PITCH_LENGTH_METRES, width=PITCH_WIDTH_METRES):
    """
    Draws standard football / futsal pitch markings.
    """
    # Outer pitch boundary
    ax.add_patch(patches.Rectangle((0, 0), length, width, fill=True, color="#2e7d32", alpha=0.9))
    ax.add_patch(patches.Rectangle((0, 0), length, width, fill=False, color="white", linewidth=2))

    # Halfway line
    ax.plot([length / 2, length / 2], [0, width], color="white", linewidth=2)

    # Center circle
    center_circle = patches.Circle((length / 2, width / 2), 6.0, fill=False, color="white", linewidth=2)
    center_spot = patches.Circle((length / 2, width / 2), 0.4, fill=True, color="white")
    ax.add_patch(center_circle)
    ax.add_patch(center_spot)

    # Penalty areas
    penalty_left = patches.Rectangle((0, width / 2 - 6), 9.0, 12.0, fill=False, color="white", linewidth=2)
    penalty_right = patches.Rectangle((length - 9.0, width / 2 - 6), 9.0, 12.0, fill=False, color="white", linewidth=2)
    ax.add_patch(penalty_left)
    ax.add_patch(penalty_right)


def plot_player_trajectories(
    positions_csv=PLAYER_POSITIONS_CSV,
    output_png=TRAJECTORIES_PLOT,
    top_n=10
):
    """
    Plots player trajectories for the longest tracked players.
    """
    print("=" * 60)
    print("GENERATING TRAJECTORY VISUALIZATIONS")
    print("=" * 60)

    if not Path(positions_csv).exists():
        print(f"Positions file not found: {positions_csv}")
        return

    df = pd.read_csv(positions_csv)
    if len(df) == 0:
        print("No position data available.")
        return

    is_pitch_coord = (df["coordinate_system"] == "pitch_world").any() and df["pitch_x"].notna().any()

    # Find longest tracks
    track_lengths = df.groupby("track_id").size().sort_values(ascending=False)
    longest_tracks = track_lengths.head(top_n).index

    fig, ax = plt.subplots(figsize=(13, 8), facecolor="#121826")
    ax.set_facecolor("#1a2234")

    # Distinct color palette
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))

    if is_pitch_coord:
        draw_futsal_pitch(ax)
        for i, track_id in enumerate(longest_tracks):
            g = df[(df["track_id"] == track_id) & df["pitch_x"].notna()].sort_values("frame")
            if len(g) > 0:
                ax.plot(
                    g["pitch_x"],
                    g["pitch_y"],
                    label=f"Track {track_id} ({len(g)} frames)",
                    color=colors[i],
                    linewidth=2.2,
                    alpha=0.85
                )
                # Mark start and end points
                ax.scatter(g["pitch_x"].iloc[0], g["pitch_y"].iloc[0], color=colors[i], marker="o", s=60, edgecolors="white")
                ax.scatter(g["pitch_x"].iloc[-1], g["pitch_y"].iloc[-1], color=colors[i], marker="X", s=80, edgecolors="white")

        ax.set_xlim(-2, PITCH_LENGTH_METRES + 2)
        ax.set_ylim(-2, PITCH_WIDTH_METRES + 2)
        ax.set_xlabel("Pitch X (Metres)", color="white", fontsize=12, fontweight="bold")
        ax.set_ylabel("Pitch Y (Metres)", color="white", fontsize=12, fontweight="bold")
        ax.set_title(f"Top {len(longest_tracks)} Player Trajectories on Calibrated Pitch [Pitch-World Coordinates]", color="white", fontsize=14, fontweight="bold", pad=15)
    else:
        for i, track_id in enumerate(longest_tracks):
            g = df[df["track_id"] == track_id].sort_values("frame")
            ax.plot(
                g["player_x"],
                g["player_y"],
                label=f"Track {track_id} ({len(g)} frames)",
                color=colors[i],
                linewidth=2.2,
                alpha=0.85
            )
            # Mark start and end
            ax.scatter(g["player_x"].iloc[0], g["player_y"].iloc[0], color=colors[i], marker="o", s=60, edgecolors="white")
            ax.scatter(g["player_x"].iloc[-1], g["player_y"].iloc[-1], color=colors[i], marker="X", s=80, edgecolors="white")

        ax.invert_yaxis()
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)
        ax.set_xlabel("Image X (Pixels)", color="white", fontsize=12, fontweight="bold")
        ax.set_ylabel("Image Y (Pixels)", color="white", fontsize=12, fontweight="bold")
        ax.set_title(f"Top {len(longest_tracks)} Player Trajectories [Image Coordinates - 1920x1080]", color="white", fontsize=14, fontweight="bold", pad=15)

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#4a5568")

    ax.legend(facecolor="#1e293b", edgecolor="#4a5568", labelcolor="white", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3, color="white")

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor())
    plt.close()

    print(f"Trajectory plot saved to: {output_png}")


def plot_gameplay_coverage(
    frames_csv=SCOREBOARD_FRAMES_CSV,
    segments_csv=SCOREBOARD_SEGMENTS_CSV,
    output_png=CAMERA_COVERAGE_PLOT
):
    """
    Plots the match coverage timeline showing active gameplay vs ads/breaks.
    """
    if not Path(frames_csv).exists() or not Path(segments_csv).exists():
        return

    frames_df = pd.read_csv(frames_csv)
    segments_df = pd.read_csv(segments_csv)

    fig, ax = plt.subplots(figsize=(13, 4), facecolor="#121826")
    ax.set_facecolor("#1a2234")

    # Plot baseline
    max_time = frames_df["video_time"].max() if len(frames_df) > 0 else 0
    ax.barh(0, max_time, height=0.4, color="#e53e3e", alpha=0.7, label="Non-Gameplay / Break / Ad")

    # Overlay gameplay segments
    for _, seg in segments_df.iterrows():
        st = float(seg["start_time"])
        dur = float(seg["duration"])
        ax.barh(0, dur, left=st, height=0.4, color="#38a169", alpha=0.95, label="Valid Gameplay" if _ == 0 else "")

    ax.set_yticks([])
    ax.set_xlabel("Video Time (Seconds)", color="white", fontsize=12)
    ax.set_title("Match Video Gameplay vs Break Breakdown (Scoreboard Filtered)", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#4a5568")

    ax.legend(facecolor="#1e293b", edgecolor="#4a5568", labelcolor="white", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Gameplay coverage plot saved to: {output_png}")


if __name__ == "__main__":
    plot_player_trajectories()
    plot_gameplay_coverage()