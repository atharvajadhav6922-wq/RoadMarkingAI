"""
src/image_processor.py

Processes individual photographs (JPG, PNG, etc.) for lane marking analysis.

Steps:
  1. Read image
  2. Extract EXIF GPS if available
  3. Apply ROI
  4. Run lane detector
  5. Annotate evidence image
  6. Return structured result

Works without GPS (graceful degradation).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.gps_manager import EXIFGPSExtractor, GPSPoint
from src.lane_detector import BaseLaneDetector, LaneDetectionResult
from src.road_roi import RoadROI, ROIConfig

logger = logging.getLogger(__name__)


@dataclass
class PhotoResult:
    """Result of processing a single photograph."""
    path: Path
    width: int = 0
    height: int = 0
    detection: Optional[LaneDetectionResult] = None
    gps_point: Optional[GPSPoint] = None
    annotated_frame: Optional[np.ndarray] = None
    error: Optional[str] = None

    @property
    def resolution(self) -> str:
        return f"{self.width} × {self.height}"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "filename": self.path.name,
            "resolution": self.resolution,
            "error": self.error,
        }
        if self.detection:
            d.update(self.detection.to_dict())
        if self.gps_point:
            d["latitude"] = self.gps_point.latitude
            d["longitude"] = self.gps_point.longitude
            d["gps_accuracy_m"] = self.gps_point.accuracy_m
            d["gps_source"] = self.gps_point.source
        return d


class ImageProcessor:
    """Processes one or more photographs."""

    def __init__(
        self,
        detector: BaseLaneDetector,
        roi_config: Optional[ROIConfig] = None,
    ):
        self.detector = detector
        self.roi = RoadROI(roi_config)
        self._exif_extractor = EXIFGPSExtractor()

    def process_file(self, path: Path) -> PhotoResult:
        """Process a single image file."""
        result = PhotoResult(path=path)
        try:
            raw_bytes = path.read_bytes()

            # Read with OpenCV for processing
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                result.error = f"Cannot decode image: {path.name}"
                return result

            result.height, result.width = frame.shape[:2]

            # Extract EXIF GPS
            result.gps_point = self._exif_extractor.extract(raw_bytes)

            # Lane detection
            result.detection = self.detector.detect_lane_markings(frame)

            # Annotate
            result.annotated_frame = self._annotate(frame, result.detection, result.gps_point)

        except Exception as exc:
            result.error = f"Processing error: {exc}"
            logger.exception(f"ImageProcessor error for {path.name}")

        return result

    def process_bytes(self, image_bytes: bytes, filename: str = "image") -> PhotoResult:
        """Process image from bytes (e.g., Streamlit upload)."""
        tmp_path = Path(filename)
        result = PhotoResult(path=tmp_path)
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                result.error = "Cannot decode image bytes"
                return result
            result.height, result.width = frame.shape[:2]
            result.gps_point = self._exif_extractor.extract(image_bytes)
            result.detection = self.detector.detect_lane_markings(frame)
            result.annotated_frame = self._annotate(frame, result.detection, result.gps_point)
        except Exception as exc:
            result.error = f"Processing error: {exc}"
        return result

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        det: LaneDetectionResult,
        gps: Optional[GPSPoint],
    ) -> np.ndarray:
        out = frame.copy()
        color = (0, 255, 0) if det.detected else (0, 0, 255)

        for seg in det.segments:
            cv2.line(out, (seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]), color, 3)

        status = "DETECTED" if det.detected else "NOT DETECTED"
        label = f"{status}  conf={det.confidence:.2f}  vis={det.visibility_score:.2f}"
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if gps:
            gps_label = f"GPS: {gps.latitude:.6f}, {gps.longitude:.6f}"
            cv2.putText(out, gps_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        if det.warning:
            cv2.putText(out, f"! {det.warning[:60]}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        return out
