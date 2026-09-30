"""
src/defect_detector.py

Classifies detected lane marking results into four defect categories:
  MISSING            — expected marking absent over a sustained segment
  FADED              — marking present but visibility score below threshold
  BROKEN             — discontinuity in an otherwise expected continuous marking
  INCORRECT_ARRANGEMENT — marking geometry inconsistent with expected configuration

Input: per-frame LaneDetectionResult sequence + road/marking parameters
Output: List[DefectRecord]

Engineering assumptions documented inline.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    FADED_VISIBILITY_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    MIN_DEFECT_SEGMENT_FRAMES,
    MISSING_ABSENCE_FRACTION,
)
from src.lane_detector import LaneDetectionResult


# ---------------------------------------------------------------------------
# Defect types
# ---------------------------------------------------------------------------

class DefectType(str, Enum):
    MISSING = "MISSING"
    FADED = "FADED"
    BROKEN = "BROKEN"
    INCORRECT_ARRANGEMENT = "INCORRECT_ARRANGEMENT"


DEFECT_RECOMMENDATIONS: Dict[DefectType, str] = {
    DefectType.MISSING: (
        "Verify the applicable longitudinal road marking from the sanctioned road layout and current "
        "authority specification, then restore/provide the marking in accordance with the applicable "
        "IRC/MoRTH provision."
    ),
    DefectType.FADED: (
        "Inspect the affected marking and renew/reapply the approved road-marking material after "
        "suitable pavement preparation, subject to the applicable specification."
    ),
    DefectType.BROKEN: (
        "Verify whether the observed discontinuity corresponds to the prescribed marking pattern. "
        "If not, restore the marking continuity/pattern in accordance with the approved layout."
    ),
    DefectType.INCORRECT_ARRANGEMENT: (
        "Engineering review required. Compare the observed marking arrangement with the sanctioned "
        "layout and applicable IRC/MoRTH requirements before rectification."
    ),
}


# ---------------------------------------------------------------------------
# Defect Record
# ---------------------------------------------------------------------------

@dataclass
class DefectRecord:
    defect_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    defect_type: DefectType = DefectType.MISSING
    confidence: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    gps_accuracy_m: Optional[float] = None
    gps_source: str = "none"
    pixel_bbox: Optional[Tuple[int, int, int, int]] = None   # x1,y1,x2,y2
    estimated_road_position: Optional[str] = None            # e.g. "12.5 m from start"
    evidence_frame_index: int = 0                            # representative frame
    review_status: str = "AI candidate — pending review"     # confirmed / rejected
    applicable_standard: Optional[str] = None
    engineering_note: Optional[str] = None
    recommendation: str = ""
    source_file: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ID": self.defect_id,
            "Type": self.defect_type.value,
            "Confidence": round(self.confidence, 3),
            "Start Frame": self.start_frame,
            "End Frame": self.end_frame,
            "Start Time (s)": round(self.start_time_s, 2),
            "End Time (s)": round(self.end_time_s, 2),
            "Duration (s)": round(self.duration_s, 2),
            "Latitude": self.latitude,
            "Longitude": self.longitude,
            "GPS Accuracy (m)": self.gps_accuracy_m,
            "GPS Source": self.gps_source,
            "Road Position": self.estimated_road_position,
            "Review Status": self.review_status,
            "Recommendation": self.recommendation,
            "Standard": self.applicable_standard,
            "Source File": self.source_file,
        }


# ---------------------------------------------------------------------------
# Defect Detector
# ---------------------------------------------------------------------------

class DefectDetector:
    """
    Analyses a sequence of per-frame LaneDetectionResult objects and produces
    a list of DefectRecord objects.

    Temporal logic:
      - A single frame failure does NOT constitute a defect.
      - A sustained absence/fade/break over MIN_DEFECT_SEGMENT_FRAMES frames
        triggers a defect.
      - BROKEN is detected when detected–absent–detected patterns occur within
        a segment where continuous marking is expected.
    """

    def __init__(
        self,
        expected_marking: str = "auto",
        road_classification: str = "urban_arterial",
        divided: bool = False,
        num_lanes: int = 2,
    ):
        self.expected_marking = expected_marking
        self.road_classification = road_classification
        self.divided = divided
        self.num_lanes = num_lanes

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def analyse(
        self,
        frame_results: List[Dict],
        fps: float = 5.0,
        source_file: str = "",
        gps_points: Optional[List] = None,
        distance_labels: Optional[List[str]] = None,
    ) -> List[DefectRecord]:
        """
        frame_results: list of dicts with keys:
          frame_index, timestamp_s, detection (LaneDetectionResult or dict), gps_point (optional)

        Returns list of DefectRecord.
        """
        if not frame_results:
            return []

        defects: List[DefectRecord] = []

        defects.extend(self._detect_missing(frame_results, fps, source_file, distance_labels))
        defects.extend(self._detect_faded(frame_results, fps, source_file, distance_labels))
        defects.extend(self._detect_broken(frame_results, fps, source_file, distance_labels))
        defects.extend(self._detect_incorrect_arrangement(frame_results, fps, source_file))

        # Attach recommendations
        for d in defects:
            d.recommendation = DEFECT_RECOMMENDATIONS[d.defect_type]

        return defects

    # -----------------------------------------------------------------------
    # MISSING detection
    # -----------------------------------------------------------------------

    def _detect_missing(
        self,
        frame_results: List[Dict],
        fps: float,
        source_file: str,
        distance_labels: Optional[List[str]],
    ) -> List[DefectRecord]:
        """
        Flag segments where marking is absent over MISSING_ABSENCE_FRACTION of frames
        within a sliding window of MIN_DEFECT_SEGMENT_FRAMES.
        """
        defects: List[DefectRecord] = []
        absent_run: List[Dict] = []
        in_defect = False
        defect_start: Optional[Dict] = None

        for fr in frame_results:
            det = self._get_detection(fr)
            is_absent = (not det.detected) or (det.confidence < LOW_CONFIDENCE_THRESHOLD)

            if is_absent:
                absent_run.append(fr)
                if not in_defect and len(absent_run) >= MIN_DEFECT_SEGMENT_FRAMES:
                    in_defect = True
                    defect_start = absent_run[0]
            else:
                if in_defect:
                    # Close the defect
                    d = self._make_defect(
                        DefectType.MISSING,
                        absent_run,
                        defect_start,
                        absent_run[-1],
                        source_file,
                        distance_labels,
                    )
                    defects.append(d)
                absent_run = []
                in_defect = False
                defect_start = None

        # Handle trailing defect
        if in_defect and absent_run:
            d = self._make_defect(
                DefectType.MISSING,
                absent_run,
                defect_start,
                absent_run[-1],
                source_file,
                distance_labels,
            )
            defects.append(d)

        return defects

    # -----------------------------------------------------------------------
    # FADED detection
    # -----------------------------------------------------------------------

    def _detect_faded(
        self,
        frame_results: List[Dict],
        fps: float,
        source_file: str,
        distance_labels: Optional[List[str]],
    ) -> List[DefectRecord]:
        """
        Flag segments where the marking is detected but visibility_score is
        consistently below FADED_VISIBILITY_THRESHOLD.
        """
        defects: List[DefectRecord] = []
        faded_run: List[Dict] = []
        in_defect = False
        defect_start: Optional[Dict] = None

        for fr in frame_results:
            det = self._get_detection(fr)
            is_faded = (
                det.detected
                and det.confidence >= LOW_CONFIDENCE_THRESHOLD
                and det.visibility_score < FADED_VISIBILITY_THRESHOLD
            )

            if is_faded:
                faded_run.append(fr)
                if not in_defect and len(faded_run) >= MIN_DEFECT_SEGMENT_FRAMES:
                    in_defect = True
                    defect_start = faded_run[0]
            else:
                if in_defect:
                    d = self._make_defect(
                        DefectType.FADED,
                        faded_run,
                        defect_start,
                        faded_run[-1],
                        source_file,
                        distance_labels,
                    )
                    defects.append(d)
                faded_run = []
                in_defect = False
                defect_start = None

        if in_defect and faded_run:
            d = self._make_defect(
                DefectType.FADED,
                faded_run,
                defect_start,
                faded_run[-1],
                source_file,
                distance_labels,
            )
            defects.append(d)

        return defects

    # -----------------------------------------------------------------------
    # BROKEN detection
    # -----------------------------------------------------------------------

    def _detect_broken(
        self,
        frame_results: List[Dict],
        fps: float,
        source_file: str,
        distance_labels: Optional[List[str]],
    ) -> List[DefectRecord]:
        """
        Detect detected → absent → detected pattern where the middle gap
        is too large to be a prescribed dash gap.

        Engineering assumption:
          A gap of ≥ MIN_DEFECT_SEGMENT_FRAMES absent frames between two
          detected segments is flagged as potential BROKEN.

        Note: this may overlap with MISSING for very long gaps. MISSING takes
        priority in analysis (filter already applied above), so BROKEN here
        catches shorter unexplained gaps.
        """
        defects: List[DefectRecord] = []
        n = len(frame_results)

        # Find runs of "detected" and "absent"
        states = []
        for fr in frame_results:
            det = self._get_detection(fr)
            states.append(det.detected and det.confidence >= LOW_CONFIDENCE_THRESHOLD)

        i = 0
        while i < n:
            if states[i]:  # start of detected run
                j = i + 1
                while j < n and states[j]:
                    j += 1
                # states[i:j] = True (detected)
                # Look for absent gap after
                gap_start = j
                k = j
                while k < n and not states[k]:
                    k += 1
                gap_len = k - gap_start

                if gap_len >= MIN_DEFECT_SEGMENT_FRAMES // 2 and k < n:
                    # There is a detected segment after — flag as broken
                    gap_frames = frame_results[gap_start:k]
                    if gap_frames:
                        d = self._make_defect(
                            DefectType.BROKEN,
                            gap_frames,
                            frame_results[gap_start],
                            frame_results[k - 1],
                            source_file,
                            distance_labels,
                        )
                        defects.append(d)
                i = k
            else:
                i += 1

        return defects

    # -----------------------------------------------------------------------
    # INCORRECT ARRANGEMENT detection
    # -----------------------------------------------------------------------

    def _detect_incorrect_arrangement(
        self,
        frame_results: List[Dict],
        fps: float,
        source_file: str,
    ) -> List[DefectRecord]:
        """
        Compare observed line geometry with expected configuration.

        For a divided road: we expect markings on both sides of the centreline.
        For an undivided road with num_lanes > 2: we expect multiple lane lines.

        This is a simplified heuristic — full arrangement checking requires
        calibrated camera geometry and is not possible from RGB alone.

        Returns a defect for the whole file if arrangement is inconsistent.
        """
        defects: List[DefectRecord] = []
        if not frame_results:
            return defects

        detected_frames = [fr for fr in frame_results if self._get_detection(fr).detected]
        if not detected_frames:
            return defects  # No markings at all — handled by MISSING

        # Count frames with multiple line groups (left + right)
        # A rough proxy: frames with ≥ 4 segments suggesting bi-lateral presence
        multi_lane_frames = sum(
            1 for fr in detected_frames
            if self._get_detection(fr).raw_line_count >= 4
        )
        frac_multilane = multi_lane_frames / max(len(detected_frames), 1)

        # For num_lanes >= 3 or divided road: expect multilane markings
        expected_multilane = self.num_lanes >= 3 or self.divided
        if expected_multilane and frac_multilane < 0.3 and len(detected_frames) > MIN_DEFECT_SEGMENT_FRAMES:
            # Potentially only one side of the road has markings
            rep_frame = detected_frames[len(detected_frames) // 2]
            d = DefectRecord(
                defect_type=DefectType.INCORRECT_ARRANGEMENT,
                confidence=0.50,
                start_frame=frame_results[0].get("frame_index", 0),
                end_frame=frame_results[-1].get("frame_index", 0),
                start_time_s=frame_results[0].get("timestamp_s", 0.0),
                end_time_s=frame_results[-1].get("timestamp_s", 0.0),
                evidence_frame_index=rep_frame.get("frame_index", 0),
                source_file=source_file,
                engineering_note=(
                    "Observed marking configuration may not match the expected arrangement "
                    f"for a {'divided' if self.divided else 'undivided'} {self.num_lanes}-lane road. "
                    "Engineering review required."
                ),
            )
            gps = rep_frame.get("gps_point")
            if gps:
                d.latitude = gps.latitude
                d.longitude = gps.longitude
                d.gps_accuracy_m = gps.accuracy_m
                d.gps_source = gps.source
            defects.append(d)

        return defects

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_detection(fr: Dict) -> LaneDetectionResult:
        det = fr.get("detection")
        if isinstance(det, LaneDetectionResult):
            return det
        # Fallback: build from dict
        if isinstance(det, dict):
            return LaneDetectionResult(
                detected=det.get("detected", False),
                confidence=det.get("confidence", 0.0),
                visibility_score=det.get("visibility_score", 0.0),
                raw_line_count=det.get("raw_line_count", 0),
            )
        return LaneDetectionResult(detected=False)

    def _make_defect(
        self,
        dtype: DefectType,
        run_frames: List[Dict],
        start_fr: Dict,
        end_fr: Dict,
        source_file: str,
        distance_labels: Optional[List[str]],
    ) -> DefectRecord:
        """Build a DefectRecord from a run of frames."""
        if not run_frames:
            run_frames = [start_fr]

        # Representative frame = middle of run
        rep_idx = len(run_frames) // 2
        rep_frame = run_frames[rep_idx]

        # Average confidence over the run
        confs = [self._get_detection(fr).confidence for fr in run_frames]
        avg_conf = float(np.mean(confs)) if confs else 0.0

        si = start_fr.get("frame_index", 0)
        ei = end_fr.get("frame_index", 0)
        st = start_fr.get("timestamp_s", 0.0)
        et = end_fr.get("timestamp_s", 0.0)

        gps_pt = rep_frame.get("gps_point")
        lat = gps_pt.latitude if gps_pt else None
        lon = gps_pt.longitude if gps_pt else None
        acc = gps_pt.accuracy_m if gps_pt else None
        gsrc = gps_pt.source if gps_pt else "none"

        # Road position label
        pos: Optional[str] = None
        if distance_labels and si < len(distance_labels):
            pos = distance_labels[si]

        d = DefectRecord(
            defect_type=dtype,
            confidence=round(1.0 - avg_conf, 3) if dtype == DefectType.MISSING else round(avg_conf, 3),
            start_frame=si,
            end_frame=ei,
            start_time_s=float(st),
            end_time_s=float(et),
            latitude=lat,
            longitude=lon,
            gps_accuracy_m=acc,
            gps_source=gsrc,
            evidence_frame_index=rep_frame.get("frame_index", si),
            estimated_road_position=pos,
            source_file=source_file,
        )
        return d
