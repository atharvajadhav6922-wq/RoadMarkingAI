"""
src/lane_detector.py

Modular lane-marking detector.

Architecture:
  BaseLaneDetector          — abstract interface
  ClassicalCVLaneDetector   — production baseline using OpenCV classical CV
  SegmentationLaneDetector  — stub for future model-based detection

The interface detect_lane_markings(frame) returns a standardised dict so
that downstream code is decoupled from the detection backend.

Indian road challenge considerations:
  - Wet pavement creates specular reflections → white-pixel approach fails
  - Dust / dirt reduces contrast
  - Glare / headlights saturate regions
  - Advertisements/markings on roadside walls create false positives
  - Partially occluded markings from vehicles

Mitigations in ClassicalCVLaneDetector:
  - HSV thresholding in both white and yellow ranges (not just brightness)
  - Morphological cleaning to remove noise
  - Aspect-ratio / angle filtering on Hough lines (road lines are near-vertical
    in forward camera perspective, or have characteristic slope angles)
  - Confidence scoring based on response strength, not binary detection
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    CANNY_HIGH,
    CANNY_LOW,
    HOUGH_MAX_LINE_GAP,
    HOUGH_MIN_LINE_LEN,
    HOUGH_RHO,
    HOUGH_THETA_DEG,
    HOUGH_THRESHOLD,
    MORPH_KERNEL_SIZE,
    WHITE_HSV_LOWER,
    WHITE_HSV_UPPER,
    YELLOW_HSV_LOWER,
    YELLOW_HSV_UPPER,
)
from src.road_roi import RoadROI, ROIConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class LaneDetectionResult:
    detected: bool
    segments: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    visibility_score: float = 0.0
    marking_color: str = "unknown"     # "white" | "yellow" | "mixed" | "unknown"
    raw_line_count: int = 0
    debug_mask: Optional[np.ndarray] = None    # binary marking mask (for evidence)
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "segments": self.segments,
            "confidence": round(self.confidence, 3),
            "visibility_score": round(self.visibility_score, 3),
            "marking_color": self.marking_color,
            "raw_line_count": self.raw_line_count,
            "warning": self.warning,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLaneDetector(ABC):
    """
    Interface for lane marking detectors.
    Subclass this to add segmentation model support.
    """

    @abstractmethod
    def detect_lane_markings(self, frame: np.ndarray) -> LaneDetectionResult:
        """
        Detect lane markings in a BGR frame.

        Returns a LaneDetectionResult with at minimum:
          detected, confidence, visibility_score, segments
        """
        ...

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        result: LaneDetectionResult,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw detected segments on a copy of the frame."""
        out = frame.copy()
        for seg in result.segments:
            x1, y1, x2, y2 = seg.get("x1", 0), seg.get("y1", 0), seg.get("x2", 0), seg.get("y2", 0)
            cv2.line(out, (x1, y1), (x2, y2), color, thickness=3)
        return out


# ---------------------------------------------------------------------------
# Classical CV detector
# ---------------------------------------------------------------------------

class ClassicalCVLaneDetector(BaseLaneDetector):
    """
    Baseline detector using:
      1. HSV thresholding for white AND yellow road markings
      2. Morphological cleaning
      3. Canny edge detection on the masked region
      4. Probabilistic Hough Line Transform
      5. Slope / angle filtering (road lines ≈ ±20–75° from vertical in forward view)
      6. Confidence = f(line count, line length, visibility strength)

    Limitations (documented):
      - Retroreflectivity cannot be measured from RGB imagery
      - Severe glare, rain, or occlusion may cause missed detections
      - Contrast estimation is a proxy for visibility only
      - Not validated to any IRC/MoRTH measurement standard
    """

    def __init__(
        self,
        roi_config: Optional[ROIConfig] = None,
        white_lower: Tuple = WHITE_HSV_LOWER,
        white_upper: Tuple = WHITE_HSV_UPPER,
        yellow_lower: Tuple = YELLOW_HSV_LOWER,
        yellow_upper: Tuple = YELLOW_HSV_UPPER,
    ):
        self.roi = RoadROI(roi_config)
        self.white_lower = np.array(white_lower, dtype=np.uint8)
        self.white_upper = np.array(white_upper, dtype=np.uint8)
        self.yellow_lower = np.array(yellow_lower, dtype=np.uint8)
        self.yellow_upper = np.array(yellow_upper, dtype=np.uint8)
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def detect_lane_markings(self, frame: np.ndarray) -> LaneDetectionResult:
        if frame is None or frame.size == 0:
            return LaneDetectionResult(detected=False, warning="Empty frame")

        h, w = frame.shape[:2]
        if h < 50 or w < 50:
            return LaneDetectionResult(detected=False, warning="Frame too small")

        # --- Apply ROI mask ---
        roi_frame = self.roi.apply(frame)

        # --- HSV thresholding ---
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Determine dominant color
        white_px = int(np.sum(white_mask > 0))
        yellow_px = int(np.sum(yellow_mask > 0))
        if white_px == 0 and yellow_px == 0:
            marking_color = "unknown"
        elif white_px > yellow_px * 2:
            marking_color = "white"
        elif yellow_px > white_px * 2:
            marking_color = "yellow"
        else:
            marking_color = "mixed"

        # --- Morphological cleaning ---
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self._morph_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self._morph_kernel)

        # --- Visibility score (fraction of expected road area with markings) ---
        roi_area = self._roi_area(h, w)
        total_marking_px = int(np.sum(cleaned > 0))
        # Visibility: normalised 0–1, clamped
        visibility_score = min(1.0, total_marking_px / max(roi_area * 0.03, 1))

        # --- Canny on the cleaned mask (improves line detection quality) ---
        edges = cv2.Canny(cleaned, CANNY_LOW, CANNY_HIGH)

        # --- Hough lines ---
        hough_theta = np.deg2rad(HOUGH_THETA_DEG)
        lines = cv2.HoughLinesP(
            edges,
            rho=HOUGH_RHO,
            theta=hough_theta,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LEN,
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )

        segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle_deg = self._line_angle_deg(x1, y1, x2, y2)
                # Road markings in forward perspective have characteristic slopes
                # Roughly: slope angle 15–75° from horizontal  (or ≈ 15–75 degrees)
                if not self._is_road_line_angle(angle_deg):
                    continue
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                segments.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "length_px": float(length),
                    "angle_deg": float(angle_deg),
                })

        # --- Confidence scoring ---
        confidence = self._compute_confidence(segments, visibility_score, h, w)

        # --- Detect dark / glare image quality issues ---
        warning = self._check_image_quality(roi_frame)

        detected = confidence > 0.15 and len(segments) > 0

        return LaneDetectionResult(
            detected=detected,
            segments=segments,
            confidence=confidence,
            visibility_score=visibility_score,
            marking_color=marking_color,
            raw_line_count=len(segments),
            debug_mask=cleaned,
            warning=warning,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _roi_area(self, h: int, w: int) -> float:
        from config.settings import (
            ROI_BOTTOM_LEFT_X_FRAC, ROI_BOTTOM_RIGHT_X_FRAC,
            ROI_BOTTOM_Y_FRAC, ROI_TOP_LEFT_X_FRAC,
            ROI_TOP_RIGHT_X_FRAC, ROI_TOP_Y_FRAC,
        )
        # Shoelace formula for trapezoid
        tl = (ROI_TOP_LEFT_X_FRAC * w,    ROI_TOP_Y_FRAC * h)
        tr = (ROI_TOP_RIGHT_X_FRAC * w,   ROI_TOP_Y_FRAC * h)
        br = (ROI_BOTTOM_RIGHT_X_FRAC * w, ROI_BOTTOM_Y_FRAC * h)
        bl = (ROI_BOTTOM_LEFT_X_FRAC * w,  ROI_BOTTOM_Y_FRAC * h)
        pts = [tl, tr, br, bl]
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    @staticmethod
    def _line_angle_deg(x1: int, y1: int, x2: int, y2: int) -> float:
        dx = x2 - x1
        dy = y2 - y1
        return abs(math.degrees(math.atan2(dy, dx + 1e-6)))

    @staticmethod
    def _is_road_line_angle(angle_deg: float) -> bool:
        """
        Filter lines by angle.
        In a forward-facing camera:
          - True road markings appear at ~15–75° from horizontal
          - Near-horizontal lines (< 15°) are horizon artefacts
          - Near-vertical lines (> 75°) may be kerbs/walls
        """
        return 15.0 <= angle_deg <= 75.0

    @staticmethod
    def _compute_confidence(
        segments: List[Dict],
        visibility_score: float,
        h: int,
        w: int,
    ) -> float:
        """
        Heuristic confidence score [0, 1].
        Combines:
          - Number of valid line segments (more = higher conf)
          - Total detected length relative to expected road width
          - Visibility score (contrast strength)
        """
        if not segments:
            return 0.0
        n = len(segments)
        total_len = sum(s["length_px"] for s in segments)
        max_expected_len = w * 2.0   # rough expectation
        length_score = min(1.0, total_len / max_expected_len)
        count_score = min(1.0, n / 10.0)
        return float(0.4 * count_score + 0.3 * length_score + 0.3 * visibility_score)

    @staticmethod
    def _check_image_quality(frame: np.ndarray) -> Optional[str]:
        """Return a warning string if the image quality is problematic."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 30:
            return "Image is very dark — detection may be unreliable"
        if mean_brightness > 225:
            return "Image may be overexposed / severe glare — detection may be unreliable"
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < 5.0:
            return "Image is blurry — detection confidence reduced"
        return None


# ---------------------------------------------------------------------------
# Segmentation model stub
# ---------------------------------------------------------------------------

class SegmentationLaneDetector(BaseLaneDetector):
    """
    Stub for future segmentation-model-based detection.

    To implement:
    1. Load your model weights in __init__
    2. Run inference in detect_lane_markings
    3. Convert the segmentation mask to LaneDetectionResult

    The interface is identical to ClassicalCVLaneDetector so no downstream
    code needs changing.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.model = None
        if model_path and model_path.exists():
            self._load_model(model_path)

    def _load_model(self, path: Path) -> None:
        """Load model weights. Implement for your chosen framework."""
        logger.info(f"SegmentationLaneDetector: model loading from {path} — not yet implemented.")

    def detect_lane_markings(self, frame: np.ndarray) -> LaneDetectionResult:
        """
        Placeholder: falls back to classical detector if no model loaded.
        Replace this with actual inference.
        """
        logger.warning("SegmentationLaneDetector: no model loaded, falling back to ClassicalCV")
        return ClassicalCVLaneDetector().detect_lane_markings(frame)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_detector(model_path: Optional[Path] = None) -> BaseLaneDetector:
    """
    Return the best available detector:
      - If a model file is provided and exists → SegmentationLaneDetector
      - Otherwise → ClassicalCVLaneDetector
    """
    if model_path and model_path.exists():
        return SegmentationLaneDetector(model_path)
    return ClassicalCVLaneDetector()

