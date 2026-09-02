"""
Pitch Calibration & Homography Transformation Module.
Supports 2D homography estimation from image coordinates to standard pitch coordinates (metres)
for both WIDE and SIDE broadcast camera angles.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PITCH_LENGTH_METRES, PITCH_WIDTH_METRES, DATA_DIR

WIDE_MATRIX_PATH = DATA_DIR / "homography_wide.npy"
SIDE_MATRIX_PATH = DATA_DIR / "homography_side.npy"


class PitchCalibrator:
    def __init__(self, pitch_length=PITCH_LENGTH_METRES, pitch_width=PITCH_WIDTH_METRES):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.h_wide = None
        self.h_side = None
        self.load_default_calibrations()

    def load_default_calibrations(self):
        """Loads saved homography matrices if available."""
        if WIDE_MATRIX_PATH.exists():
            try:
                self.h_wide = np.load(str(WIDE_MATRIX_PATH))
            except Exception:
                self.h_wide = None
        else:
            # Approximate standard homography for game2 WIDE broadcast angle
            # Pitch points: (0,0), (60,0), (60,35), (0,35)
            # Image points corresponding to pitch corners in game2 WIDE view
            img_pts_wide = np.array([
                [310, 480],    # Top-Left corner / touchline
                [1610, 480],   # Top-Right corner / touchline
                [1840, 980],   # Bottom-Right corner
                [80, 980]      # Bottom-Left corner
            ], dtype=np.float32)

            pitch_pts_wide = np.array([
                [0.0, 0.0],
                [self.pitch_length, 0.0],
                [self.pitch_length, self.pitch_width],
                [0.0, self.pitch_width]
            ], dtype=np.float32)

            H, _ = cv2.findHomography(img_pts_wide, pitch_pts_wide)
            self.h_wide = H
            if H is not None:
                np.save(str(WIDE_MATRIX_PATH), H)

    def image_to_pitch(self, x, y, view="WIDE"):
        """
        Transforms image pixel coordinate (x, y) to real-world pitch coordinate (metres).
        Returns (pitch_x, pitch_y) or (None, None) if out of bounds or uncalibrated.
        """
        H = self.h_wide if view == "WIDE" else self.h_side
        if H is None:
            return None, None

        pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
        try:
            transformed = cv2.perspectiveTransform(pts, H)
            px = float(transformed[0][0][0])
            py = float(transformed[0][0][1])

            # Sanity clamp to pitch bounds + buffer
            if -5.0 <= px <= self.pitch_length + 5.0 and -5.0 <= py <= self.pitch_width + 5.0:
                return round(np.clip(px, 0.0, self.pitch_length), 2), round(np.clip(py, 0.0, self.pitch_width), 2)
            return None, None
        except Exception:
            return None, None


if __name__ == "__main__":
    calibrator = PitchCalibrator()
    print("Pitch Calibrator Initialized.")
    test_pts = [(960, 700), (400, 550), (1500, 550)]
    for x, y in test_pts:
        px, py = calibrator.image_to_pitch(x, y, view="WIDE")
        print(f"Pixel ({x}, {y}) -> Pitch ({px}m, {py}m)")