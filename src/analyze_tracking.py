import pandas as pd

INPUT = "data/tracking.csv"

df = pd.read_csv(INPUT)

print("=" * 60)
print("TRACKING ANALYSIS")
print("=" * 60)

print(f"Total records: {len(df):,}")
print(f"Frames: {df['frame'].min()} - {df['frame'].max()}")
print(f"Unique track IDs: {df['track_id'].nunique()}")

# ----------------------------------------
# TRACK LENGTH
# ----------------------------------------

track_stats = (
    df.groupby("track_id")
    .agg(
        frames_tracked=("frame", "count"),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        avg_confidence=("confidence", "mean"),
    )
    .sort_values("frames_tracked", ascending=False)
)

print()
print("TOP 30 LONGEST TRACKS")
print("-" * 60)

print(
    track_stats.head(30).to_string(
        formatters={
            "avg_confidence": "{:.3f}".format
        }
    )
)

# ----------------------------------------
# TRACK DURATION
# ----------------------------------------

print()
print("TRACK LENGTH DISTRIBUTION")
print("-" * 60)

print(
    track_stats["frames_tracked"]
    .describe()
)

# ----------------------------------------
# VERY SHORT TRACKS
# ----------------------------------------

short_tracks = (track_stats["frames_tracked"] <= 10).sum()

print()
print(f"Tracks lasting <= 10 frames: {short_tracks}")
print(f"Tracks lasting > 100 frames: {(track_stats['frames_tracked'] > 100).sum()}")
print(f"Tracks lasting > 500 frames: {(track_stats['frames_tracked'] > 500).sum()}")

print("=" * 60)