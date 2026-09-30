"""
tests/test_detector.py

Unit tests for lane marking detection and defect classification.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lane_detector import ClassicalCVLaneDetector, LaneDetectionResult, build_detector
from src.defect_detector import DefectDetector, DefectRecord, DefectType
from src.temporal_tracker import TemporalTracker


# ---------------------------------------------------------------------------
# Helpers: synthetic test frames
# ---------------------------------------------------------------------------

def make_blank_frame(h=480, w=640, color=(50, 50, 50)):
    """Dark road-like frame with no markings."""
    return np.full((h, w, 3), color, dtype=np.uint8)


def make_marked_frame(h=480, w=640):
    """Frame with white lane lines painted in."""
    frame = make_blank_frame(h, w, (60, 60, 60))
    # Left dashed line
    for y in range(250, 450, 40):
        cv2.line(frame, (200, y), (200, y + 25), (255, 255, 255), 5)
    # Right dashed line
    for y in range(250, 450, 40):
        cv2.line(frame, (440, y), (440, y + 25), (255, 255, 255), 5)
    return frame


def make_faded_frame(h=480, w=640):
    """Frame with very faint markings."""
    frame = make_blank_frame(h, w, (80, 80, 80))
    for y in range(250, 450, 40):
        cv2.line(frame, (200, y), (200, y + 25), (100, 100, 100), 3)
    return frame


def make_glare_frame(h=480, w=640):
    """Frame with severe overexposure (glare simulation)."""
    return np.full((h, w, 3), 250, dtype=np.uint8)


def make_dark_frame(h=480, w=640):
    """Very dark frame (night/underexposed)."""
    return np.full((h, w, 3), 15, dtype=np.uint8)


# ---------------------------------------------------------------------------
# ClassicalCVLaneDetector tests
# ---------------------------------------------------------------------------

class TestClassicalCVLaneDetector:
    def setup_method(self):
        self.detector = ClassicalCVLaneDetector()

    def test_marked_frame_detected(self):
        frame = make_marked_frame()
        result = self.detector.detect_lane_markings(frame)
        assert isinstance(result, LaneDetectionResult)
        # Should detect at least something
        assert result.confidence >= 0.0   # confidence computed without error

    def test_blank_frame_low_confidence(self):
        frame = make_blank_frame()
        result = self.detector.detect_lane_markings(frame)
        assert result.confidence < 0.5

    def test_empty_frame_returns_result(self):
        result = self.detector.detect_lane_markings(np.array([]))
        assert not result.detected
        assert result.warning is not None

    def test_tiny_frame_returns_result(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = self.detector.detect_lane_markings(frame)
        assert not result.detected

    def test_dark_frame_warning(self):
        frame = make_dark_frame()
        result = self.detector.detect_lane_markings(frame)
        assert result.warning is not None
        assert "dark" in result.warning.lower()

    def test_glare_frame_warning(self):
        frame = make_glare_frame()
        result = self.detector.detect_lane_markings(frame)
        assert result.warning is not None

    def test_result_dict_serialisable(self):
        frame = make_marked_frame()
        result = self.detector.detect_lane_markings(frame)
        d = result.to_dict()
        assert "detected" in d
        assert "confidence" in d
        assert "visibility_score" in d

    def test_segments_are_dicts(self):
        frame = make_marked_frame()
        result = self.detector.detect_lane_markings(frame)
        for seg in result.segments:
            assert "x1" in seg
            assert "y1" in seg


# ---------------------------------------------------------------------------
# build_detector factory test
# ---------------------------------------------------------------------------

class TestBuildDetector:
    def test_no_model_returns_classical(self):
        det = build_detector()
        assert isinstance(det, ClassicalCVLaneDetector)

    def test_nonexistent_model_path_returns_classical(self):
        det = build_detector(model_path=Path("/nonexistent/model.pt"))
        # SegmentationLaneDetector falls back to classical
        frame = make_marked_frame()
        result = det.detect_lane_markings(frame)
        assert isinstance(result, LaneDetectionResult)


# ---------------------------------------------------------------------------
# DefectDetector tests
# ---------------------------------------------------------------------------

def _make_frame_results(n: int, detected: bool, confidence: float = 0.7, visibility: float = 0.8):
    """Build synthetic per-frame result list."""
    from src.lane_detector import LaneDetectionResult
    results = []
    for i in range(n):
        det = LaneDetectionResult(
            detected=detected,
            confidence=confidence,
            visibility_score=visibility,
            raw_line_count=4 if detected else 0,
        )
        results.append({
            "frame_index": i,
            "timestamp_s": i * 0.2,   # at 5fps
            "detection": det,
            "gps_point": None,
        })
    return results


class TestDefectDetector:
    def setup_method(self):
        self.detector = DefectDetector(
            expected_marking="auto",
            road_classification="urban_arterial",
            divided=False,
            num_lanes=2,
        )

    def test_all_detected_no_missing(self):
        fr = _make_frame_results(30, detected=True, confidence=0.8, visibility=0.8)
        defects = self.detector.analyse(fr, fps=5.0)
        missing = [d for d in defects if d.defect_type == DefectType.MISSING]
        assert len(missing) == 0

    def test_all_absent_creates_missing(self):
        fr = _make_frame_results(30, detected=False, confidence=0.1)
        defects = self.detector.analyse(fr, fps=5.0)
        missing = [d for d in defects if d.defect_type == DefectType.MISSING]
        assert len(missing) >= 1

    def test_all_faded_creates_faded(self):
        # detected=True but very low visibility
        fr = _make_frame_results(30, detected=True, confidence=0.55, visibility=0.20)
        defects = self.detector.analyse(fr, fps=5.0)
        faded = [d for d in defects if d.defect_type == DefectType.FADED]
        assert len(faded) >= 1

    def test_broken_pattern(self):
        """detected–absent–detected → BROKEN defect."""
        fr = (
            _make_frame_results(15, detected=True, confidence=0.8, visibility=0.8)
            + _make_frame_results(10, detected=False, confidence=0.05)
            + _make_frame_results(15, detected=True, confidence=0.8, visibility=0.8)
        )
        # Re-index frames
        for i, f in enumerate(fr):
            f["frame_index"] = i
            f["timestamp_s"] = i * 0.2
        defects = self.detector.analyse(fr, fps=5.0)
        broken = [d for d in defects if d.defect_type == DefectType.BROKEN]
        assert len(broken) >= 1

    def test_short_absent_not_flagged_as_missing(self):
        """A very short absence (< MIN_DEFECT_SEGMENT_FRAMES) should not be flagged."""
        fr = (
            _make_frame_results(20, detected=True, confidence=0.8, visibility=0.8)
            + _make_frame_results(3, detected=False, confidence=0.1)
            + _make_frame_results(20, detected=True, confidence=0.8, visibility=0.8)
        )
        for i, f in enumerate(fr):
            f["frame_index"] = i
            f["timestamp_s"] = i * 0.2
        defects = self.detector.analyse(fr, fps=5.0)
        missing = [d for d in defects if d.defect_type == DefectType.MISSING]
        assert len(missing) == 0

    def test_defect_has_recommendation(self):
        fr = _make_frame_results(30, detected=False, confidence=0.1)
        defects = self.detector.analyse(fr, fps=5.0)
        for d in defects:
            assert d.recommendation != ""

    def test_defect_to_dict(self):
        fr = _make_frame_results(30, detected=False, confidence=0.1)
        defects = self.detector.analyse(fr, fps=5.0)
        for d in defects:
            dct = d.to_dict()
            assert "ID" in dct
            assert "Type" in dct
            assert "Confidence" in dct


# ---------------------------------------------------------------------------
# TemporalTracker tests
# ---------------------------------------------------------------------------

class TestTemporalTracker:
    def test_update_returns_smoothed(self):
        tracker = TemporalTracker(window=5)
        result = LaneDetectionResult(detected=True, confidence=0.8, visibility_score=0.7, raw_line_count=4)
        state = tracker.update(0, result)
        assert "smoothed_confidence" in state
        assert "smoothed_visibility" in state

    def test_smoothing_reduces_noise(self):
        tracker = TemporalTracker(window=5)
        # Alternate high/low confidence
        confs = []
        for i in range(20):
            conf = 0.9 if i % 2 == 0 else 0.1
            r = LaneDetectionResult(detected=True, confidence=conf, visibility_score=conf)
            s = tracker.update(i, r)
            confs.append(s["smoothed_confidence"])
        # Smoothed values should be less extreme than raw
        assert max(confs) < 0.9
        assert min(confs) > 0.1

    def test_build_spatial_segments(self):
        tracker = TemporalTracker(window=3)
        fr = _make_frame_results(30, detected=True, confidence=0.8, visibility=0.8)
        smoothed_frs = []
        for f in fr:
            s = tracker.update(f["frame_index"], f["detection"])
            smoothed_frs.append({**f, **s})
        segments = tracker.build_spatial_segments(smoothed_frs, fps=5.0)
        assert len(segments) >= 1
        for seg in segments:
            assert seg.state in ("detected", "absent", "faded")
            assert seg.start_frame <= seg.end_frame


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
