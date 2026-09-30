"""
tests/test_report.py

Unit tests for PDF report generation and supporting subsystems.
"""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.defect_detector import DefectRecord, DefectType, DEFECT_RECOMMENDATIONS
from src.road_classification import RoadParameters, ROAD_CLASSES, EXPECTED_MARKINGS
from src.standards_engine import StandardsEngine, StandardsResult
from src.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def make_road_params(**kwargs) -> RoadParameters:
    defaults = dict(
        road_name="Test Road",
        authority="Test Authority",
        inspection_date=str(date.today()),
        inspector_name="Test Inspector",
        road_class="urban_arterial",
        road_class_display="Urban Arterial",
        divided=False,
        num_lanes=2,
        carriageway_width_m=7.0,
        lane_width_m=3.5,
        shoulder_width_m=1.0,
        pavement_type="asphalt",
        inspection_direction="North-bound",
        expected_marking="auto",
        expected_marking_display="Auto / Hybrid",
    )
    defaults.update(kwargs)
    return RoadParameters(**defaults)


def make_defect(dtype: DefectType = DefectType.MISSING, **kwargs) -> DefectRecord:
    defaults = dict(
        defect_type=dtype,
        confidence=0.72,
        start_frame=10,
        end_frame=25,
        start_time_s=2.0,
        end_time_s=5.0,
        latitude=12.9716,
        longitude=77.5942,
        gps_accuracy_m=4.5,
        gps_source="csv",
        source_file="test_video.mp4",
        review_status="AI candidate — pending review",
        recommendation=DEFECT_RECOMMENDATIONS[dtype],
    )
    defaults.update(kwargs)
    return DefectRecord(**defaults)


# ---------------------------------------------------------------------------
# RoadParameters tests
# ---------------------------------------------------------------------------

class TestRoadParameters:
    def test_from_ui_inputs(self):
        ui = {
            "road_name": "MG Road",
            "authority": "BBMP",
            "inspection_date": "2025-01-15",
            "inspector_name": "John Doe",
            "road_class_display": "Urban Arterial",
            "divided": False,
            "num_lanes": 4,
            "carriageway_width_m": 14.0,
            "lane_width_m": 3.5,
            "shoulder_width_m": 1.0,
            "pavement_type_display": "Asphalt / Bituminous",
            "inspection_direction": "North",
            "expected_marking_display": "Centre Line",
        }
        rp = RoadParameters.from_ui_inputs(ui)
        assert rp.road_name == "MG Road"
        assert rp.road_class == "urban_arterial"
        assert rp.num_lanes == 4
        assert rp.expected_marking == "centre_line"

    def test_geometry_label(self):
        rp = make_road_params(divided=True, num_lanes=4)
        label = rp.geometry_label
        assert "Divided" in label
        assert "4" in label

    def test_road_classes_complete(self):
        for k, v in ROAD_CLASSES.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    def test_expected_markings_complete(self):
        for k, v in EXPECTED_MARKINGS.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# StandardsEngine tests
# ---------------------------------------------------------------------------

class TestStandardsEngine:
    def setup_method(self):
        self.engine = StandardsEngine()

    def test_engine_loads_rules(self):
        rules = self.engine.get_all_references()
        assert len(rules) > 0

    def test_query_returns_result(self):
        result = self.engine.query(
            road_class="urban_arterial",
            marking_type="centre_line",
            divided=False,
        )
        assert isinstance(result, StandardsResult)
        assert result.verification_required is True

    def test_query_has_disclaimer(self):
        result = self.engine.query("urban_arterial")
        assert "verification" in result.disclaimer.lower()

    def test_unknown_road_class_has_fallback(self):
        result = self.engine.query("completely_unknown_class")
        assert isinstance(result, StandardsResult)
        # engineering_note should still be set
        assert result.engineering_note != ""

    def test_verification_always_required(self):
        for rule in self.engine.get_all_references():
            assert rule.get("verification_required") is True, (
                f"Rule {rule.get('id')} missing verification_required=true"
            )

    def test_no_fabricated_dimensions(self):
        """All rules must have source_document set."""
        for rule in self.engine.get_all_references():
            assert rule.get("source_document"), f"Rule {rule.get('id')} missing source_document"


# ---------------------------------------------------------------------------
# DefectRecord tests
# ---------------------------------------------------------------------------

class TestDefectRecord:
    def test_duration(self):
        d = make_defect(start_time_s=2.0, end_time_s=8.0)
        assert d.duration_s == pytest.approx(6.0)

    def test_to_dict_keys(self):
        d = make_defect()
        dct = d.to_dict()
        required_keys = ["ID", "Type", "Confidence", "Latitude", "Longitude", "Review Status"]
        for k in required_keys:
            assert k in dct, f"Missing key: {k}"

    def test_recommendation_set(self):
        for dtype in DefectType:
            rec = DEFECT_RECOMMENDATIONS[dtype]
            assert len(rec) > 20  # non-trivial recommendation
            assert "IRC" in rec or "specify" in rec.lower() or "layout" in rec.lower() or "specification" in rec.lower()

    def test_defect_id_unique(self):
        ids = {make_defect().defect_id for _ in range(50)}
        assert len(ids) == 50  # all unique


# ---------------------------------------------------------------------------
# ReportGenerator tests
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def test_generate_returns_path(self, tmp_path, monkeypatch):
        # Monkey-patch REPORTS_DIR to tmp_path
        import config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "REPORTS_DIR", tmp_path)

        rp = make_road_params()
        defects = [make_defect(DefectType.MISSING), make_defect(DefectType.FADED)]
        standards = StandardsEngine().query("urban_arterial")
        gen = ReportGenerator()
        path = gen.generate(
            road_params=rp,
            defects=defects,
            standards_result=standards,
            gps_summary={"available": False, "mode": "none"},
            session_id="test-session",
            source_files=["video.mp4"],
            total_frames=100,
        )
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.stat().st_size > 0

    def test_generate_bytes_returns_bytes(self, tmp_path, monkeypatch):
        import config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "REPORTS_DIR", tmp_path)

        rp = make_road_params()
        gen = ReportGenerator()
        pdf_bytes = gen.generate_bytes(
            road_params=rp,
            defects=[make_defect()],
            standards_result=None,
            gps_summary={"available": False, "mode": "none"},
            session_id="test-bytes",
            source_files=["photo.jpg"],
            total_frames=1,
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        # PDF magic bytes
        assert pdf_bytes[:4] == b"%PDF"

    def test_report_with_no_defects(self, tmp_path, monkeypatch):
        import config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "REPORTS_DIR", tmp_path)

        rp = make_road_params()
        gen = ReportGenerator()
        path = gen.generate(
            road_params=rp,
            defects=[],
            standards_result=None,
            gps_summary={"available": False, "mode": "none"},
            session_id="test-empty",
            source_files=[],
            total_frames=0,
        )
        assert path.exists()

    def test_report_with_gps(self, tmp_path, monkeypatch):
        import config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "REPORTS_DIR", tmp_path)

        rp = make_road_params()
        gen = ReportGenerator()
        path = gen.generate(
            road_params=rp,
            defects=[make_defect()],
            standards_result=None,
            gps_summary={
                "available": True,
                "mode": "csv",
                "point_count": 120,
                "total_distance_m": 450.0,
                "avg_accuracy": "±4.5 m",
                "source": "csv",
            },
            session_id="test-gps",
            source_files=["video.mp4"],
            total_frames=150,
        )
        assert path.exists()


# ---------------------------------------------------------------------------
# Input classification tests
# ---------------------------------------------------------------------------

class TestInputClassification:
    def test_image_extensions(self):
        from src.input_manager import classify_extension
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            assert classify_extension(f"file{ext}") == "image", f"Failed for {ext}"

    def test_video_extensions(self):
        from src.input_manager import classify_extension
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".m4v"]:
            assert classify_extension(f"file{ext}") == "video", f"Failed for {ext}"

    def test_unsupported_returns_none(self):
        from src.input_manager import classify_extension
        assert classify_extension("document.pdf") is None
        assert classify_extension("data.xlsx") is None
        assert classify_extension("script.py") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
