"""
Scoreboard OCR Module.
Extracts match clock, team names, and score from the scoreboard region
across detected gameplay segments using Tesseract OCR.
"""
import cv2
import csv
import re
import sys
from pathlib import Path
import pytesseract

# Set utf-8 encoding for standard output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add src to path if executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    VIDEO_PATH,
    SCOREBOARD_SEGMENTS_CSV,
    SCOREBOARD_DATA_CSV,
    SCOREBOARD_ROI,
    TESSERACT_EXE_PATH,
    OCR_FRAME_STEP
)

# Configure Tesseract path
if Path(TESSERACT_EXE_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE_PATH)


def preprocess_scoreboard_crop(roi):
    """
    Preprocess scoreboard ROI for optimal OCR recognition.
    """
    if roi is None or roi.size == 0:
        return None

    # Upscale 3x for clearer character boundaries
    enlarged = cv2.resize(
        roi,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )
    
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Binary thresholding for high contrast text
    _, thresh = cv2.threshold(
        blurred,
        150,
        255,
        cv2.THRESH_BINARY
    )
    return thresh


def parse_scoreboard_text(text):
    """
    Parses OCR raw text to extract match clock, team scores, and team names.
    Returns (clock, score, raw_text).
    """
    text_clean = text.strip()
    
    # 1. Match Clock Pattern: MM:SS
    clock_matches = re.findall(r"\b(\d{1,2}):(\d{2})\b", text_clean)
    clock = None
    if clock_matches:
        mins, secs = clock_matches[0]
        mins, secs = int(mins), int(secs)
        if 0 <= mins <= 130 and 0 <= secs <= 59:
            clock = f"{mins}:{secs:02d}"

    # 2. Score Pattern: [0-9] [0-9] or [0-9] - [0-9]
    score_matches = re.findall(r"\b(\d{1,2})\s*[-:]?\s*(\d{1,2})\b", text_clean)
    score = None
    if score_matches:
        for s1, s2 in score_matches:
            s1_int, s2_int = int(s1), int(s2)
            if s1_int <= 30 and s2_int <= 30:
                score = f"{s1_int}-{s2_int}"
                break

    return clock, score, text_clean


def run_scoreboard_ocr(
    video_path=VIDEO_PATH,
    segments_csv=SCOREBOARD_SEGMENTS_CSV,
    output_csv=SCOREBOARD_DATA_CSV,
    step=OCR_FRAME_STEP,
    max_samples_per_segment=20
):
    """
    Runs OCR on frames across all detected scoreboard gameplay segments.
    """
    print("=" * 60)
    print("SCOREBOARD OCR EXTRACTION")
    print("=" * 60)

    try:
        tess_ver = pytesseract.get_tesseract_version()
        print(f"Tesseract Version: {tess_ver}")
    except Exception as e:
        print(f"Warning: Tesseract initialization error: {e}")

    if not Path(segments_csv).exists():
        raise FileNotFoundError(f"Could not find segments file: {segments_csv}. Run scoreboard detector first.")

    with open(segments_csv, "r", encoding="utf-8") as f:
        segments = list(csv.DictReader(f))

    print(f"Total gameplay segments to process: {len(segments)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    results = []
    x1, y1, x2, y2 = SCOREBOARD_ROI

    for seg_idx, seg in enumerate(segments, start=1):
        start_frame = int(seg["start_frame"])
        end_frame = int(seg["end_frame"])
        seg_duration = float(seg["duration"])

        print(f"Processing Segment {seg_idx:02d}/{len(segments):02d}: Frames {start_frame:,} -> {end_frame:,} ({seg_duration:.1f}s)")

        # Sample frames evenly across the segment
        frame_span = end_frame - start_frame
        if frame_span <= 0:
            frame_indices = [start_frame]
        else:
            n_samples = min(max_samples_per_segment, max(1, frame_span // step))
            sample_step = max(step, frame_span // n_samples)
            frame_indices = list(range(start_frame, end_frame + 1, sample_step))

        for f_num in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
            ret, frame = cap.read()
            if not ret:
                break

            crop = frame[y1:y2, x1:x2]
            thresh = preprocess_scoreboard_crop(crop)
            
            raw_text = ""
            clock = None
            score = None
            confidence = 0.0

            if thresh is not None:
                try:
                    ocr_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config="--psm 6")
                    raw_text = " ".join([word for word in ocr_data['text'] if word.strip()])
                    conf_vals = [float(c) for c in ocr_data['conf'] if str(c).isnumeric() and float(c) >= 0]
                    confidence = (sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0
                    clock, score, _ = parse_scoreboard_text(raw_text)
                except Exception:
                    raw_text = ""

            video_time = f_num / fps
            results.append({
                "segment": seg_idx,
                "frame": f_num,
                "video_time": round(video_time, 3),
                "match_time": clock if clock else "",
                "score": score if score else "",
                "raw_ocr": raw_text.replace("\n", " "),
                "ocr_confidence": round(confidence, 2)
            })

    cap.release()

    # Save to CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["segment", "frame", "video_time", "match_time", "score", "raw_ocr", "ocr_confidence"]
        )
        writer.writeheader()
        writer.writerows(results)

    success_clocks = sum(1 for r in results if r["match_time"])
    success_scores = sum(1 for r in results if r["score"])
    print()
    print("=" * 60)
    print("OCR EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total frames sampled:       {len(results):,}")
    print(f"Scores detected:            {success_scores:,}")
    print(f"Match clocks detected:      {success_clocks:,}")
    print(f"Saved to:                   {output_csv}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_scoreboard_ocr()