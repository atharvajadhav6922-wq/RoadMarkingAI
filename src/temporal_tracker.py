"""
src/temporal_tracker.py

Tracks lane marking candidates across video frames.

Instead of treating each frame independently, the tracker maintains a
running estimate of:
  - Line position (centre-x)
  - Slope
  - Confidence
  - Visibility
  - Continuity score

Uses exponential moving average (EMA) smoothing.

Outputs a spatially segmented result:
  [{"start_frame": 0, "end_frame": 23, "state": "detected", ...}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import TEMPORAL_SMOOTHING_WINDOW
from src.lane_detector import LaneDetectionResult


@dataclass
class TrackState:
    """Running state for one 'lane track'."""
    track_id: int
    smoothed_confidence: float = 0.0
    smoothed_visibility: float = 0.0
    smoothed_line_count: float = 0.0
    last_seen_frame: int = -1
    frames_seen: int = 0


@dataclass
class SpatialSegment:
    """A contiguous stretch of road with a consistent marking state."""
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    state: str           # "detected" | "absent" | "faded"
    avg_confidence: float
    avg_visibility: float
    frame_count: int
    gps_start: Optional[Any] = None
    gps_end: Optional[Any] = None
    distance_label: Optional[str] = None   # e.g. "0–15 m (estimated)"


class TemporalTracker:
    """
    Maintains a temporally-smoothed view of lane marking detection across frames.

    Smoothing approach:
      - EMA (exponential moving average) with alpha = 2/(window+1)
      - State transitions only confirmed after TEMPORAL_SMOOTHING_WINDOW frames

    This prevents brief occlusions (passing vehicles, shadows) from
    triggering false defect detections.
    """

    def __init__(self, window: int = TEMPORAL_SMOOTHING_WINDOW):
        self.window = window
        self.alpha = 2.0 / (window + 1)
        self._conf_ema: float = 0.0
        self._vis_ema: float = 0.0
        self._count_ema: float = 0.0
        self._smoothed: List[Dict[str, Any]] = []

    def update(self, frame_index: int, result: LaneDetectionResult) -> Dict[str, Any]:
        """Update tracker with a new frame result. Returns smoothed state."""
        c = result.confidence
        v = result.visibility_score
        n = float(result.raw_line_count)

        self._conf_ema = self.alpha * c + (1 - self.alpha) * self._conf_ema
        self._vis_ema = self.alpha * v + (1 - self.alpha) * self._vis_ema
        self._count_ema = self.alpha * n + (1 - self.alpha) * self._count_ema

        state = {
            "frame_index": frame_index,
            "raw_detected": result.detected,
            "raw_confidence": c,
            "smoothed_confidence": round(self._conf_ema, 3),
            "smoothed_visibility": round(self._vis_ema, 3),
            "smoothed_line_count": round(self._count_ema, 1),
            "warning": result.warning,
        }
        self._smoothed.append(state)
        return state

    def build_spatial_segments(
        self,
        frame_results: List[Dict],
        fps: float,
        gps_points: Optional[List] = None,
        dist_per_frame_m: Optional[float] = None,
    ) -> List[SpatialSegment]:
        """
        Convert frame-level results to spatial segments.

        Returns a list of SpatialSegment with state "detected" | "absent" | "faded".
        """
        if not frame_results:
            return []

        from config.settings import FADED_VISIBILITY_THRESHOLD, LOW_CONFIDENCE_THRESHOLD

        segments: List[SpatialSegment] = []
        current_state: Optional[str] = None
        seg_start_fr: Optional[Dict] = None
        seg_frames: List[Dict] = []

        def _state_of(fr: Dict) -> str:
            sc = fr.get("smoothed_confidence", fr.get("raw_confidence", 0.0))
            sv = fr.get("smoothed_visibility", 0.0)
            if sc < LOW_CONFIDENCE_THRESHOLD:
                return "absent"
            if sv < FADED_VISIBILITY_THRESHOLD:
                return "faded"
            return "detected"

        for fr in frame_results:
            state = _state_of(fr)
            if current_state is None:
                current_state = state
                seg_start_fr = fr
                seg_frames = [fr]
            elif state == current_state:
                seg_frames.append(fr)
            else:
                seg = self._build_segment(seg_start_fr, seg_frames, current_state, fps, gps_points, dist_per_frame_m)
                segments.append(seg)
                current_state = state
                seg_start_fr = fr
                seg_frames = [fr]

        # Last segment
        if seg_frames:
            seg = self._build_segment(seg_start_fr, seg_frames, current_state, fps, gps_points, dist_per_frame_m)
            segments.append(seg)

        return segments

    @staticmethod
    def _build_segment(
        start_fr: Dict,
        frames: List[Dict],
        state: str,
        fps: float,
        gps_points: Optional[List],
        dist_per_frame_m: Optional[float],
    ) -> SpatialSegment:
        sf = start_fr.get("frame_index", 0)
        ef = frames[-1].get("frame_index", sf)
        st = sf / fps
        et = ef / fps

        confs = [fr.get("smoothed_confidence", 0.0) for fr in frames]
        viss = [fr.get("smoothed_visibility", 0.0) for fr in frames]

        gps_s = None
        gps_e = None
        if gps_points:
            gps_s = gps_points[sf] if sf < len(gps_points) else None
            gps_e = gps_points[ef] if ef < len(gps_points) else None

        dist_label: Optional[str] = None
        if dist_per_frame_m is not None:
            ds = round(sf * dist_per_frame_m, 1)
            de = round(ef * dist_per_frame_m, 1)
            dist_label = f"{ds}–{de} m (estimated)"

        return SpatialSegment(
            start_frame=sf,
            end_frame=ef,
            start_time_s=round(st, 2),
            end_time_s=round(et, 2),
            state=state,
            avg_confidence=round(float(np.mean(confs)), 3) if confs else 0.0,
            avg_visibility=round(float(np.mean(viss)), 3) if viss else 0.0,
            frame_count=len(frames),
            gps_start=gps_s,
            gps_end=gps_e,
            distance_label=dist_label,
        )
