"""
src/video_processor.py

Reads video files, extracts metadata, and samples frames at a configurable rate.

Per sampled frame:
  - Detect road region (ROI)
  - Run lane detector
  - Associate GPS timestamp if available
  - Store frame evidence

Returns a list of per-frame result dicts for downstream defect analysis.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import DEFAULT_SAMPLE_FPS, MAX_SAMPLE_FPS, MIN_SAMPLE_FPS
from src.gps_manager import GPSTrack, GPSPoint
from src.lane_detector import BaseLaneDetector, LaneDetectionResult
from src.road_roi import RoadROI, ROIConfig
from src.temporal_tracker import TemporalTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

@dataclass
class VideoMetadata:
    path: Path
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_s: float = 0.0
    codec: str = "unknown"
    error: Optional[str] = None

    @property
    def resolution(self) -> str:
        return f"{self.width} × {self.height}"

    @property
    def duration_str(self) -> str:
        m = int(self.duration_s // 60)
        s = self.duration_s % 60
        return f"{m:02d}:{s:05.2f}"


def read_video_metadata(path: Path) -> VideoMetadata:
    """Extract metadata from a video file using OpenCV."""
    meta = VideoMetadata(path=path)
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            meta.error = f"Cannot open video: {path.name}"
            return meta
        meta.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        meta.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        meta.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if meta.fps > 0:
            meta.duration_s = meta.frame_count / meta.fps
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        meta.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()
    except Exception as exc:
        meta.error = str(exc)
    return meta


# ---------------------------------------------------------------------------
# Frame sampler
# ---------------------------------------------------------------------------

class VideoProcessor:
    """
    Samples frames from a video, runs lane detection, and associates GPS.

    sample_fps: how many frames per second to process (default 5)
    """

    def __init__(
        self,
        detector: BaseLaneDetector,
        roi_config: Optional[ROIConfig] = None,
        sample_fps: float = DEFAULT_SAMPLE_FPS,
    ):
        self.detector = detector
        self.roi = RoadROI(roi_config)
        self.sample_fps = max(MIN_SAMPLE_FPS, min(MAX_SAMPLE_FPS, sample_fps))
        self.tracker = TemporalTracker()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def process(
        self,
        video_path: Path,
        gps_track: Optional[GPSTrack] = None,
        video_start_datetime: Optional[datetime] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[VideoMetadata, List[Dict[str, Any]]]:
        """
        Process a video file.

        Returns:
            (VideoMetadata, list of per-frame result dicts)

        Each frame result dict contains:
            frame_index, timestamp_s, detection, gps_point, raw_frame, annotated_frame,
            smoothed_confidence, smoothed_visibility
        """
        meta = read_video_metadata(video_path)
        if meta.error:
            logger.error(f"VideoProcessor: {meta.error}")
            return meta, []

        frame_results: List[Dict[str, Any]] = []
        video_fps = meta.fps if meta.fps > 0 else 25.0
        frame_step = max(1, int(round(video_fps / self.sample_fps)))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            meta.error = f"Cannot open video for processing: {video_path.name}"
            return meta, []

        try:
            frame_idx = 0
            processed = 0
            total_frames = meta.frame_count or 1

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    timestamp_s = frame_idx / video_fps
                    gps_pt = self._get_gps(gps_track, timestamp_s, video_start_datetime)

                    # Run detection
                    try:
                        result: LaneDetectionResult = self.detector.detect_lane_markings(frame)
                    except Exception as exc:
                        logger.warning(f"Detection error frame {frame_idx}: {exc}")
                        result = LaneDetectionResult(detected=False, warning=str(exc))

                    # Temporal tracking
                    smoothed = self.tracker.update(frame_idx, result)

                    # Annotate frame for evidence
                    annotated = self._annotate(frame, result)

                    fr_data: Dict[str, Any] = {
                        "frame_index": frame_idx,
                        "timestamp_s": timestamp_s,
                        "detection": result,
                        "gps_point": gps_pt,
                        "raw_frame": frame.copy(),
                        "annotated_frame": annotated,
                        **smoothed,
                    }
                    frame_results.append(fr_data)
                    processed += 1

                    if progress_callback and frame_idx % (frame_step * 10) == 0:
                        pct = frame_idx / total_frames
                        progress_callback(pct, f"Frame {frame_idx}/{total_frames}")

                frame_idx += 1

        finally:
            cap.release()

        if progress_callback:
            progress_callback(1.0, f"Done — {len(frame_results)} frames analysed")

        return meta, frame_results

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_gps(
        track: Optional[GPSTrack],
        timestamp_s: float,
        video_start: Optional[datetime],
    ) -> Optional[GPSPoint]:
        if track is None or not track.available:
            return None
        if video_start is None:
            return None
        ts = video_start + timedelta(seconds=timestamp_s)
        return track.interpolated_point(ts)

    @staticmethod
    def _annotate(frame: np.ndarray, result: LaneDetectionResult) -> np.ndarray:
        """Draw detection overlay on frame."""
        out = frame.copy()
        color = (0, 255, 0) if result.detected else (0, 0, 255)

        for seg in result.segments:
            cv2.line(
                out,
                (seg["x1"], seg["y1"]),
                (seg["x2"], seg["y2"]),
                color,
                thickness=3,
            )

        # Info overlay
        label = f"{'DETECTED' if result.detected else 'NOT DETECTED'} | conf={result.confidence:.2f} | vis={result.visibility_score:.2f}"
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if result.warning:
            cv2.putText(out, f"WARN: {result.warning[:60]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        return out
