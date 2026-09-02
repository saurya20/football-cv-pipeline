"""
Player Detection & Tracking Module.
Runs YOLOv11 + BotSORT tracking strictly on valid gameplay segments,
with spatial masking of broadcast overlays and reaction boxes.
"""
import cv2
import csv
import sys
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    VIDEO_PATH,
    YOLO_MODEL_PATH,
    SCOREBOARD_SEGMENTS_CSV,
    TRACKING_CSV,
    REACTION_BOX_ROI,
    SCOREBOARD_ROI,
    MIN_CONFIDENCE,
    TRACKER_TYPE
)


def apply_spatial_mask(frame):
    """
    Masks out the top-right reaction box / broadcast logos and
    top-left scoreboard banner so YOLO doesn't detect commentator faces or graphics as players.
    """
    masked = frame.copy()
    
    # Mask top-right reaction box
    rx1, ry1, rx2, ry2 = REACTION_BOX_ROI
    masked[ry1:ry2, rx1:rx2] = 0
    
    # Mask top-left scoreboard
    sx1, sy1, sx2, sy2 = SCOREBOARD_ROI
    masked[sy1:sy2, sx1:sx2] = 0
    
    return masked


def run_player_tracking(
    video_path=VIDEO_PATH,
    segments_csv=SCOREBOARD_SEGMENTS_CSV,
    output_csv=TRACKING_CSV,
    max_segments=None,
    frame_step=1,
    conf_thresh=MIN_CONFIDENCE
):
    """
    Extracts player tracking across valid gameplay segments.
    """
    print("=" * 60)
    print("PLAYER DETECTION & TRACKING (YOLOv11 + BotSORT)")
    print("=" * 60)
    print(f"Video Source:       {video_path}")
    print(f"Model:              {YOLO_MODEL_PATH}")
    print(f"Segments Source:    {segments_csv}")
    print(f"Confidence Thresh:  {conf_thresh}")
    print(f"Frame Step:         {frame_step}")
    print()

    if not Path(segments_csv).exists():
        raise FileNotFoundError(f"Scoreboard segments file not found: {segments_csv}. Please run scoreboard detector first.")

    segments_df = pd.read_csv(segments_csv)
    if len(segments_df) == 0:
        print("No valid gameplay segments found to track.")
        return

    if max_segments is not None and max_segments > 0:
        segments_df = segments_df.head(max_segments)

    print(f"Tracking across {len(segments_df)} gameplay segments...")

    model = YOLO(str(YOLO_MODEL_PATH))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "timestamp",
            "track_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence"
        ])

        total_detections = 0
        unique_tracks = set()

        for seg_idx, (_, seg) in enumerate(segments_df.iterrows(), start=1):
            start_frame = int(seg["start_frame"])
            end_frame = int(seg["end_frame"])
            seg_duration = float(seg["duration"])

            print(f"\n--- Segment {seg_idx}/{len(segments_df)}: Frames {start_frame:,} -> {end_frame:,} ({seg_duration:.1f}s) ---")

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            seg_detections = 0

            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (current_frame - start_frame) % frame_step != 0:
                    current_frame += 1
                    continue

                # Apply spatial mask to hide reaction box & scoreboard graphics
                masked_frame = apply_spatial_mask(frame)

                # Run tracker on person class (cls=0)
                results = model.track(
                    source=masked_frame,
                    classes=[0],
                    conf=conf_thresh,
                    tracker=TRACKER_TYPE,
                    persist=True,
                    verbose=False
                )

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        for i in range(len(boxes)):
                            conf = float(boxes.conf[i])
                            if conf < conf_thresh:
                                continue

                            track_id = int(boxes.id[i]) if boxes.id is not None else -1
                            if track_id >= 0:
                                unique_tracks.add(track_id)

                            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                            timestamp = round(current_frame / fps, 3)

                            writer.writerow([
                                current_frame,
                                timestamp,
                                track_id,
                                round(x1, 2),
                                round(y1, 2),
                                round(x2, 2),
                                round(y2, 2),
                                round(conf, 4)
                            ])
                            seg_detections += 1
                            total_detections += 1

                if (current_frame - start_frame) % (30 * frame_step) == 0:
                    print(f"  Frame {current_frame:,} | Segment Detections: {seg_detections:,} | Total Tracks: {len(unique_tracks):,}")

                current_frame += 1

    cap.release()

    print()
    print("=" * 60)
    print("TRACKING EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total Detections:     {total_detections:,}")
    print(f"Unique Track IDs:     {len(unique_tracks):,}")
    print(f"Saved to:             {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    run_player_tracking()