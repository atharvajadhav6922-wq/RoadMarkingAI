"""
config/settings.py
Application-wide configuration constants and defaults.
"""

import os
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
GPS_DIR = DATA_DIR / "gps"

# Ensure runtime directories exist
for _d in (UPLOAD_DIR, PROCESSED_DIR, REPORTS_DIR, GPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Video processing
# ------------------------------------------------------------------
DEFAULT_SAMPLE_FPS: float = 5.0          # frames per second to sample
MAX_SAMPLE_FPS: float = 30.0
MIN_SAMPLE_FPS: float = 1.0

# ------------------------------------------------------------------
# Road ROI — trapezoidal mask (fractions of frame height/width)
# These are defaults for a forward-facing vehicle camera.
# Top of trapezoid sits at ~45 % of frame height (horizon).
# ------------------------------------------------------------------
ROI_TOP_Y_FRAC: float = 0.45    # horizon line as fraction of frame height
ROI_BOTTOM_Y_FRAC: float = 0.95 # bottom of ROI
ROI_TOP_LEFT_X_FRAC: float = 0.35
ROI_TOP_RIGHT_X_FRAC: float = 0.65
ROI_BOTTOM_LEFT_X_FRAC: float = 0.05
ROI_BOTTOM_RIGHT_X_FRAC: float = 0.95

# ------------------------------------------------------------------
# Lane detection (classical CV)
# ------------------------------------------------------------------
# HSV thresholds for white markings
WHITE_HSV_LOWER = (0, 0, 160)
WHITE_HSV_UPPER = (180, 60, 255)

# HSV thresholds for yellow markings
YELLOW_HSV_LOWER = (15, 60, 100)
YELLOW_HSV_UPPER = (35, 255, 255)

CANNY_LOW: int = 50
CANNY_HIGH: int = 150
HOUGH_RHO: int = 1
HOUGH_THETA_DEG: float = 1.0
HOUGH_THRESHOLD: int = 30
HOUGH_MIN_LINE_LEN: int = 40
HOUGH_MAX_LINE_GAP: int = 25

# Morphological kernel size
MORPH_KERNEL_SIZE: int = 3

# ------------------------------------------------------------------
# Defect detection thresholds
# ------------------------------------------------------------------
# Minimum visibility score below which a marking is considered faded
FADED_VISIBILITY_THRESHOLD: float = 0.45

# Fraction of frames in a segment that must show absence for "missing"
MISSING_ABSENCE_FRACTION: float = 0.75

# Minimum segment length (metres or frames) to classify a defect
MIN_DEFECT_SEGMENT_FRAMES: int = 8   # at 5 fps → ~1.6 s travel

# Confidence threshold below which a detection is flagged for review
LOW_CONFIDENCE_THRESHOLD: float = 0.40

# ------------------------------------------------------------------
# Temporal tracking
# ------------------------------------------------------------------
TEMPORAL_SMOOTHING_WINDOW: int = 5   # frames

# ------------------------------------------------------------------
# GPS
# ------------------------------------------------------------------
# Columns accepted from phone GPS CSV (case-insensitive matching used)
GPS_CSV_POSSIBLE_TIMESTAMP_COLS = ["timestamp", "time", "datetime", "date_time"]
GPS_CSV_POSSIBLE_LAT_COLS = ["latitude", "lat"]
GPS_CSV_POSSIBLE_LON_COLS = ["longitude", "lon", "lng", "long"]
GPS_CSV_POSSIBLE_ACC_COLS = ["accuracy", "acc", "hdop"]

# Max allowed GPS interpolation gap (seconds) — beyond this, warn user
GPS_MAX_INTERPOLATION_GAP_S: float = 10.0

# Approximate Earth radius for Haversine distance
EARTH_RADIUS_M: float = 6_371_000.0

# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------
REPORT_VERSION: str = "1.0"
ORGANISATION_NAME: str = "RoadMarkingAI"

DISCLAIMER_TEXT: str = (
    "AI-ASSISTED INSPECTION OUTPUT\n\n"
    "This report is intended to support field inspection and engineering review only. "
    "Computer-vision results may contain false positives and false negatives, "
    "particularly under wet pavement, glare, poor visibility, occlusion, traffic "
    "and adverse lighting conditions.\n\n"
    "Final determination of road-marking requirements and rectification shall be made "
    "by the responsible authority / qualified engineer using the current applicable "
    "IRC / MoRTH provisions, sanctioned drawings and project specifications.\n\n"
    "This report does NOT constitute a legal notice of non-compliance and shall NOT "
    "be used as such without independent engineering verification."
)

# ------------------------------------------------------------------
# Supported file types
# ------------------------------------------------------------------
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
