"""
Generate Marked Video Clip and Tracking Screenshots for game2.mp4.
Renders player tracking markers, bounding boxes, and track IDs on live gameplay footage.
"""
import cv2
from pathlib import Path
from ultralytics import YOLO

# Paths
VIDEO_PATH = "data/videos/game2.mp4"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIDEO_PATH = OUTPUT_DIR / "game2_tracked_clip.mp4"

# Settings
START_FRAME = 6000     # ~100s into game2 (Segment 3, active wide tactical gameplay)
NUM_FRAMES = 360       # 360 frames = ~6 seconds at 60 FPS (fast, high quality)
CONF_THRESH = 0.30

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open source video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

# Prepare VideoWriter (mp4v codec)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_VIDEO_PATH), fourcc, fps, (width, height))

print(f"Rendering marked game2 video clip: {NUM_FRAMES} frames starting at frame {START_FRAME}...")

representative_screenshot_saved = False
screenshot_path = SCREENSHOT_DIR / "game2_player_tracking_markers.png"

for i in range(NUM_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break

    # Mask reaction box (1400, 0, 1920, 350) and scoreboard (40, 50, 650, 145)
    masked_frame = frame.copy()
    masked_frame[0:350, 1400:1920] = 0
    masked_frame[50:145, 40:650] = 0

    # Track players (class 0 = person)
    results = model.track(
        source=masked_frame,
        classes=[0],
        conf=CONF_THRESH,
        tracker="botsort.yaml",
        persist=True,
        verbose=False
    )

    # Plot bounding boxes, labels, and track IDs onto original unmasked frame
    annotated_frame = frame.copy()

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None

            # Color styling
            color = (0, 255, 127) if track_id is not None else (0, 165, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Label text
            label = f"ID: {track_id}" if track_id is not None else f"Player {conf:.2f}"
            
            # Label banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1 + 3, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Foot ground anchor point
            foot_x = int((x1 + x2) / 2)
            foot_y = y2
            cv2.circle(annotated_frame, (foot_x, foot_y), 4, (0, 0, 255), -1)

    # Save high-res screenshot around frame 60 (well established tracks)
    if i == 60 and not representative_screenshot_saved:
        cv2.imwrite(str(screenshot_path), annotated_frame)
        representative_screenshot_saved = True
        print(f"Saved marked screenshot: {screenshot_path}")

    writer.write(annotated_frame)

    if (i + 1) % 60 == 0:
        print(f"Processed {i + 1}/{NUM_FRAMES} frames...")

cap.release()
writer.release()
print(f"Marked video clip created: {OUT_VIDEO_PATH}")
