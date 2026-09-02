"""
Master Pipeline Runner for Football Computer Vision Analytics.
Executes all pipeline stages in sequence or resumes from existing checkpoints.
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    VIDEO_PATH,
    SCOREBOARD_SEGMENTS_CSV,
    SCOREBOARD_DATA_CSV,
    CAMERA_SEGMENTS_CSV,
    TRACKING_CSV,
    CLEAN_TRACKING_CSV,
    PLAYER_POSITIONS_CSV,
    PLAYER_STATISTICS_CSV,
    TRAJECTORIES_PLOT
)
from src.scoreboard_detector import detect_scoreboard
from src.scoreboard_ocr import run_scoreboard_ocr
from src.camera_classification import classify_gameplay_segments
from src.extract_tracking import run_player_tracking
from src.clean_tracking import clean_tracking_data
from src.calculate_positions import compute_player_positions
from src.calculate_statistics import calculate_movement_statistics
from src.visualize_tracks import plot_player_trajectories, plot_gameplay_coverage
from src.validate_pipeline import validate_pipeline


def main():
    parser = argparse.ArgumentParser(description="Football CV Analytics Pipeline Runner")
    parser.add_argument("--force", action="store_true", help="Force re-run all pipeline stages")
    parser.add_argument("--skip-tracking", action="store_true", help="Skip player detection and tracking stage")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip scoreboard OCR extraction")
    parser.add_argument("--max-segments", type=int, default=None, help="Limit tracking to first N gameplay segments")
    parser.add_argument("--frame-step", type=int, default=2, help="Sampling step for player tracking inside segments")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 70)
    print("★ FOOTBALL CV ANALYTICS PIPELINE ★")
    print("=" * 70)
    print(f"Target Video: {VIDEO_PATH}")
    print(f"Force Mode:   {args.force}")
    print()

    # Stage 1: Scoreboard Detection
    print("\n>>> [STAGE 1/8] SCOREBOARD & GAMEPLAY DETECTION")
    if args.force or not SCOREBOARD_SEGMENTS_CSV.exists():
        detect_scoreboard(video_path=VIDEO_PATH)
    else:
        print(f"Checkpointed: {SCOREBOARD_SEGMENTS_CSV} already exists. (Use --force to re-run)")

    # Stage 2: Scoreboard OCR
    print("\n>>> [STAGE 2/8] SCOREBOARD OCR EXTRACTION")
    if not args.skip_ocr:
        if args.force or not SCOREBOARD_DATA_CSV.exists():
            run_scoreboard_ocr(video_path=VIDEO_PATH, segments_csv=SCOREBOARD_SEGMENTS_CSV)
        else:
            print(f"Checkpointed: {SCOREBOARD_DATA_CSV} already exists.")
    else:
        print("Skipping OCR as requested (--skip-ocr).")

    # Stage 3: Camera View Classification
    print("\n>>> [STAGE 3/8] CAMERA VIEW CLASSIFICATION (WIDE / SIDE / OTHER)")
    if args.force or not CAMERA_SEGMENTS_CSV.exists():
        classify_gameplay_segments(video_path=VIDEO_PATH, segments_csv=SCOREBOARD_SEGMENTS_CSV)
    else:
        print(f"Checkpointed: {CAMERA_SEGMENTS_CSV} already exists.")

    # Stage 4: Player Detection & Tracking
    print("\n>>> [STAGE 4/8] PLAYER DETECTION & TRACKING (YOLOv11 + BotSORT)")
    if not args.skip_tracking:
        if args.force or not TRACKING_CSV.exists():
            run_player_tracking(
                video_path=VIDEO_PATH,
                segments_csv=SCOREBOARD_SEGMENTS_CSV,
                output_csv=TRACKING_CSV,
                max_segments=args.max_segments,
                frame_step=args.frame_step
            )
        else:
            print(f"Checkpointed: {TRACKING_CSV} already exists.")
    else:
        print("Skipping tracking stage as requested (--skip-tracking).")

    # Stage 5: Track Cleaning
    print("\n>>> [STAGE 5/8] TRACK CLEANING & FILTERING")
    if args.force or not CLEAN_TRACKING_CSV.exists():
        clean_tracking_data(tracking_csv=TRACKING_CSV, segments_csv=SCOREBOARD_SEGMENTS_CSV)
    else:
        print(f"Checkpointed: {CLEAN_TRACKING_CSV} already exists.")

    # Stage 6: Player Positions & Coordinate Estimation
    print("\n>>> [STAGE 6/8] PLAYER POSITION & COORDINATE ESTIMATION")
    if args.force or not PLAYER_POSITIONS_CSV.exists():
        compute_player_positions(input_csv=CLEAN_TRACKING_CSV, camera_csv=CAMERA_SEGMENTS_CSV)
    else:
        print(f"Checkpointed: {PLAYER_POSITIONS_CSV} already exists.")

    # Stage 7: Movement Statistics & Player Identity
    print("\n>>> [STAGE 7/8] MOVEMENT STATISTICS & LEADERBOARDS")
    if args.force or not PLAYER_STATISTICS_CSV.exists():
        calculate_movement_statistics(positions_csv=PLAYER_POSITIONS_CSV)
    else:
        print(f"Checkpointed: {PLAYER_STATISTICS_CSV} already exists.")

    # Stage 8: Trajectory & Timeline Visualizations
    print("\n>>> [STAGE 8/8] VISUALIZATIONS")
    plot_player_trajectories(positions_csv=PLAYER_POSITIONS_CSV)
    plot_gameplay_coverage()

    # Final Validation
    print("\n>>> [FINAL AUDIT] PIPELINE DATA VALIDATION")
    validate_pipeline()

    elapsed = time.time() - start_time
    print(f"\nPipeline run completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("Launch Dashboard: streamlit run dashboard.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
