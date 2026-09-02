"""
Configuration settings for the Football Computer Vision Pipeline.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Video Input
VIDEO_PATH = DATA_DIR / "videos" / "game2.mp4"
FALLBACK_GAME1_PATH = DATA_DIR / "videos" / "game1.mp4"

# Detection & Tracking Model
YOLO_MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"

# Scoreboard ROI (x1, y1, x2, y2) for 1920x1080 game2.mp4
# Top-left region containing team names, emblem, and score
SCOREBOARD_ROI = (40, 50, 650, 145)

# Reaction Box / Broadcast Logo Mask Region (x1, y1, x2, y2)
# Top-right region to exclude from player detection
REACTION_BOX_ROI = (1400, 0, 1920, 350)

# Sampling and Detection Settings
FRAME_STEP = 12             # ~5 checks per second at ~59.94 FPS
MIN_SEGMENT_CHECKS = 4      # Minimum consecutive positive checks to form a valid segment (~0.8s)
MIN_SEGMENT_DURATION_SEC = 2.0  # Filter out transient flash segments shorter than 2s

# Tesseract OCR Settings
TESSERACT_EXE_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_FRAME_STEP = 30         # OCR sample step inside valid gameplay segments (~2 samples/sec)

# Player Tracking & Cleaning Thresholds
MIN_CONFIDENCE = 0.30       # Minimum detection confidence
MIN_TRACK_FRAMES = 15       # Minimum frames for a valid persistent track
TRACKER_TYPE = "botsort.yaml"

# Coordinate System & Field Dimensions (Standard Futsal / 7-a-side pitch in meters)
PITCH_LENGTH_METRES = 60.0  # Length of pitch in meters
PITCH_WIDTH_METRES = 35.0   # Width of pitch in meters

# Output CSV Files
SCOREBOARD_FRAMES_CSV = DATA_DIR / "scoreboard_frames.csv"
SCOREBOARD_SEGMENTS_CSV = DATA_DIR / "scoreboard_segments.csv"
SCOREBOARD_DATA_CSV = DATA_DIR / "scoreboard_data.csv"
CAMERA_SEGMENTS_CSV = DATA_DIR / "camera_segments.csv"
TRACKING_CSV = DATA_DIR / "tracking.csv"
CLEAN_TRACKING_CSV = DATA_DIR / "clean_tracking.csv"
PLAYER_POSITIONS_CSV = DATA_DIR / "player_positions.csv"
PLAYER_STATISTICS_CSV = DATA_DIR / "player_statistics.csv"
PLAYERS_CSV = DATA_DIR / "players.csv"

# Visualization Outputs
TRAJECTORIES_PLOT = OUTPUTS_DIR / "player_trajectories.png"
CAMERA_COVERAGE_PLOT = OUTPUTS_DIR / "gameplay_coverage.png"
