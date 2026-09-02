"""
Scoreboard and Gameplay Segment Detector.
Detects presence of the match scoreboard in broadcast footage to identify
valid active gameplay segments, excluding advertisements, replays, and non-gameplay scenes.
"""
import cv2
import csv
import sys
from pathlib import Path

# Add src to path if executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    VIDEO_PATH,
    SCOREBOARD_ROI,
    FRAME_STEP,
    MIN_SEGMENT_CHECKS,
    SCOREBOARD_FRAMES_CSV,
    SCOREBOARD_SEGMENTS_CSV,
    OUTPUTS_DIR
)


def scoreboard_present(frame, roi=SCOREBOARD_ROI):
    """
    Returns True if the scoreboard region contains the match scoreboard banner.
    Uses multi-cue color, saturation, and contrast verification.
    """
    x1, y1, x2, y2 = roi
    h, w = frame.shape[:2]
    
    # Boundary clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Saturated badge pixels (orange Panda FC and purple Joga FC badges)
    sat = hsv[:, :, 1]
    sat_ratio = (sat > 65).mean()
    
    # Neon green emblem check (H: 35-85, S: 80+, V: 80+)
    green_mask = (
        (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) &
        (hsv[:, :, 1] >= 80) &
        (hsv[:, :, 2] >= 80)
    )
    green_ratio = green_mask.mean()
    
    # Dark central background bar check (V < 60)
    val = hsv[:, :, 2]
    dark_ratio = (val < 60).mean()
    
    # Scoreboard is present when either:
    # 1. Green emblem is visible + some saturation/dark background
    # 2. Strong saturation + dark banner background
    is_present = (
        (green_ratio >= 0.005 and (sat_ratio >= 0.08 or dark_ratio >= 0.15)) or
        (sat_ratio >= 0.12 and dark_ratio >= 0.20)
    )
    return bool(is_present)


def detect_scoreboard(video_path=VIDEO_PATH, frame_step=FRAME_STEP, save_debug_images=True):
    """
    Scans the video frame-by-frame sparsely to identify scoreboard-active frames
    and continuous gameplay segments.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 60)
    print("SCOREBOARD & GAMEPLAY DETECTOR")
    print("=" * 60)
    print(f"Video Path:        {video_path}")
    print(f"Video FPS:         {fps:.2f}")
    print(f"Total frames:      {total_frames:,}")
    print(f"Sampling Step:     every {frame_step} frames (~{fps/frame_step:.1f} fps)")
    print(f"Scoreboard ROI:    {SCOREBOARD_ROI}")
    print()

    debug_dir = OUTPUTS_DIR / "scoreboard_debug"
    if save_debug_images:
        debug_dir.mkdir(parents=True, exist_ok=True)

    detections = []
    detected_count = 0
    frame_number = 0
    saved_debug_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_step != 0:
            frame_number += 1
            continue

        detected = scoreboard_present(frame)
        video_time = frame_number / fps

        detections.append({
            "frame": frame_number,
            "video_time": round(video_time, 3),
            "scoreboard_present": int(detected)
        })

        if detected:
            detected_count += 1
            if save_debug_images and saved_debug_count < 10 and (detected_count % 50 == 1):
                # Save a few representative debug frames
                x1, y1, x2, y2 = SCOREBOARD_ROI
                debug_img = frame.copy()
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    debug_img,
                    f"SCOREBOARD DETECTED (t={video_time:.1f}s)",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                cv2.imwrite(str(debug_dir / f"frame_{frame_number:06d}.jpg"), debug_img)
                saved_debug_count += 1

        if len(detections) % 1000 == 0:
            print(f"Processed {frame_number:,} / {total_frames:,} frames ({(frame_number/total_frames)*100:.1f}%) | Detections: {detected_count:,}")

        frame_number += 1

    cap.release()

    # Save frame-level results
    SCOREBOARD_FRAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SCOREBOARD_FRAMES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "video_time", "scoreboard_present"]
        )
        writer.writeheader()
        writer.writerows(detections)

    # Group into continuous segments
    segments = []
    current_start = None
    current_end = None
    current_checks = 0

    for det in detections:
        f_num = det["frame"]
        present = det["scoreboard_present"]

        if present:
            if current_start is None:
                current_start = f_num
                current_end = f_num
                current_checks = 1
            else:
                current_end = f_num
                current_checks += 1
        else:
            if current_start is not None:
                if current_checks >= MIN_SEGMENT_CHECKS:
                    duration = (current_end - current_start) / fps
                    segments.append({
                        "segment": len(segments) + 1,
                        "start_frame": current_start,
                        "end_frame": current_end,
                        "start_time": round(current_start / fps, 3),
                        "end_time": round(current_end / fps, 3),
                        "duration": round(duration, 3),
                        "checks": current_checks
                    })
                current_start = None
                current_end = None
                current_checks = 0

    if current_start is not None and current_checks >= MIN_SEGMENT_CHECKS:
        duration = (current_end - current_start) / fps
        segments.append({
            "segment": len(segments) + 1,
            "start_frame": current_start,
            "end_frame": current_end,
            "start_time": round(current_start / fps, 3),
            "end_time": round(current_end / fps, 3),
            "duration": round(duration, 3),
            "checks": current_checks
        })

    # Save segments
    with open(SCOREBOARD_SEGMENTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["segment", "start_frame", "end_frame", "start_time", "end_time", "duration", "checks"]
        )
        writer.writeheader()
        writer.writerows(segments)

    # Print summary
    processed_frames = len(detections)
    det_pct = (detected_count / processed_frames * 100) if processed_frames > 0 else 0
    total_gameplay_time = sum(s["duration"] for s in segments)
    total_video_time = total_frames / fps

    print()
    print("=" * 60)
    print("SCOREBOARD DETECTION RESULTS")
    print("=" * 60)
    print(f"Frames processed:         {processed_frames:,}")
    print(f"Scoreboard detections:    {detected_count:,} ({det_pct:.1f}%)")
    print(f"Identified segments:      {len(segments)}")
    print(f"Total gameplay duration:  {total_gameplay_time:.1f}s ({total_gameplay_time/60:.1f} min)")
    print(f"Total video duration:     {total_video_time:.1f}s ({total_video_time/60:.1f} min)")
    print(f"Gameplay coverage:        {(total_gameplay_time/total_video_time)*100:.1f}%")
    print(f"Frame CSV:                {SCOREBOARD_FRAMES_CSV}")
    print(f"Segments CSV:             {SCOREBOARD_SEGMENTS_CSV}")
    print()
    print("SEGMENT SUMMARY:")
    print("-" * 60)
    for s in segments[:15]:
        print(f"Segment {s['segment']:02d}: Frames {s['start_frame']:6d} → {s['end_frame']:6d} | {s['start_time']:7.2f}s → {s['end_time']:7.2f}s ({s['duration']:6.2f}s)")
    if len(segments) > 15:
        print(f"... and {len(segments) - 15} more segments.")
    print("=" * 60)

    return segments


if __name__ == "__main__":
    detect_scoreboard()