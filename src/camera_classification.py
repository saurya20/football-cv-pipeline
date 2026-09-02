"""
Camera View Classifier.
Classifies gameplay segments into WIDE, SIDE, or OTHER camera angles
to ensure appropriate coordinate transformation and analytics models are applied.
"""
import cv2
import csv
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    VIDEO_PATH,
    SCOREBOARD_SEGMENTS_CSV,
    CAMERA_SEGMENTS_CSV
)


def classify_frame_view(frame):
    """
    Classifies a single frame into WIDE, SIDE, or OTHER based on pitch coverage,
    ground color distribution, and field geometry.
    """
    if frame is None or frame.size == 0:
        return "OTHER", 0.0

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green pitch HSV range
    # H: 30-85, S: 40-255, V: 40-255
    pitch_mask = (
        (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 85) &
        (hsv[:, :, 1] >= 35) &
        (hsv[:, :, 2] >= 35)
    )

    # Calculate pitch coverage in different regions
    top_region = pitch_mask[:int(h * 0.35), :]
    mid_region = pitch_mask[int(h * 0.35):int(h * 0.70), :]
    bottom_region = pitch_mask[int(h * 0.70):, :]

    total_pitch_ratio = pitch_mask.mean()
    mid_pitch_ratio = mid_region.mean()
    bottom_pitch_ratio = bottom_region.mean()

    # Detect line features (white lines on grass)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=10)
    num_lines = len(lines) if lines is not None else 0

    # Classification Rules
    # WIDE view: Vast majority of middle & lower screen is pitch (>55%), distinct field lines visible
    if total_pitch_ratio >= 0.45 and bottom_pitch_ratio >= 0.65:
        if num_lines >= 8 or mid_pitch_ratio >= 0.60:
            return "WIDE", round(min(0.95, total_pitch_ratio + 0.3), 2)
        else:
            return "SIDE", 0.80

    # SIDE view: Substantial pitch visible (>30%), but tighter angle or sideline visible
    elif total_pitch_ratio >= 0.25 and bottom_pitch_ratio >= 0.40:
        return "SIDE", 0.75

    # OTHER: Close-up, crowd, reaction box, replay transition
    else:
        return "OTHER", 0.85


def classify_gameplay_segments(video_path=VIDEO_PATH, segments_csv=SCOREBOARD_SEGMENTS_CSV, output_csv=CAMERA_SEGMENTS_CSV, sample_frames_per_seg=5):
    """
    Classifies all valid scoreboard gameplay segments into WIDE, SIDE, or OTHER.
    """
    print("=" * 60)
    print("CAMERA VIEW CLASSIFICATION")
    print("=" * 60)

    if not Path(segments_csv).exists():
        raise FileNotFoundError(f"Segments file not found: {segments_csv}")

    with open(segments_csv, "r", encoding="utf-8") as f:
        segments = list(csv.DictReader(f))

    if not segments:
        print("No gameplay segments found to classify.")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    classified_segments = []

    for seg_idx, seg in enumerate(segments, start=1):
        seg_id = int(seg["segment"]) if "segment" in seg else seg_idx
        start_frame = int(seg["start_frame"])
        end_frame = int(seg["end_frame"])
        start_time = float(seg["start_time"])
        end_time = float(seg["end_time"])
        duration = float(seg["duration"])

        # Sample frames across the segment
        step = max(1, (end_frame - start_frame) // (sample_frames_per_seg + 1))
        sample_frame_indices = [start_frame + (i + 1) * step for i in range(sample_frames_per_seg)]

        votes = {"WIDE": 0, "SIDE": 0, "OTHER": 0}
        confidences = []

        for f_idx in sample_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                view, conf = classify_frame_view(frame)
                votes[view] += 1
                confidences.append(conf)

        # Majority vote
        majority_view = max(votes, key=votes.get)
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.5

        classified_segments.append({
            "segment": seg_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "camera_view": majority_view,
            "confidence": round(avg_conf, 2)
        })

    cap.release()

    # Save to CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["segment", "start_frame", "end_frame", "start_time", "end_time", "duration", "camera_view", "confidence"]
        )
        writer.writeheader()
        writer.writerows(classified_segments)

    # Print summary
    wide_count = sum(1 for s in classified_segments if s["camera_view"] == "WIDE")
    side_count = sum(1 for s in classified_segments if s["camera_view"] == "SIDE")
    other_count = sum(1 for s in classified_segments if s["camera_view"] == "OTHER")

    print(f"Classified {len(classified_segments)} segments:")
    print(f"  - WIDE View:   {wide_count} segments")
    print(f"  - SIDE View:   {side_count} segments")
    print(f"  - OTHER View:  {other_count} segments")
    print(f"Saved to: {output_csv}")
    print("=" * 60)

    return classified_segments


if __name__ == "__main__":
    classify_gameplay_segments()
