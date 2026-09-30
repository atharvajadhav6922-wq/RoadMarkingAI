"""
tests/test_gps.py

Unit tests for GPS CSV parsing, EXIF extraction, distance calculation,
and GPS interpolation.
"""

import io
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gps_manager import (
    GPSCSVParser,
    GPSPoint,
    GPSTrack,
    EXIFGPSExtractor,
    GPSManager,
    haversine_distance,
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SAMPLE_CSV_STANDARD = b"""timestamp,latitude,longitude,accuracy
2025-01-15T08:00:00Z,12.971600,77.594200,4.5
2025-01-15T08:00:05Z,12.971650,77.594250,3.8
2025-01-15T08:00:10Z,12.971700,77.594300,5.1
"""

SAMPLE_CSV_ALTERNATE_COLS = b"""time,lat,lon,hdop
2025-01-15 08:00:00,12.971600,77.594200,2.1
2025-01-15 08:00:05,12.971650,77.594250,1.9
"""

SAMPLE_CSV_BAD = b"garbage,columns\nhello,world\n"

SAMPLE_CSV_INVALID_COORDS = b"""timestamp,latitude,longitude
2025-01-15T08:00:00Z,0.000000,0.000000
2025-01-15T08:00:05Z,91.0,77.5
"""


# ---------------------------------------------------------------------------
# Haversine distance tests
# ---------------------------------------------------------------------------

class TestHaversineDistance:
    def test_zero_distance(self):
        p = GPSPoint(timestamp=None, latitude=12.97, longitude=77.59)
        assert haversine_distance(p, p) == pytest.approx(0.0, abs=0.001)

    def test_known_distance(self):
        # Approximate distance between two points ~560 m apart
        p1 = GPSPoint(timestamp=None, latitude=12.9716, longitude=77.5942)
        p2 = GPSPoint(timestamp=None, latitude=12.9766, longitude=77.5942)
        dist = haversine_distance(p1, p2)
        # ~0.005 degrees lat ≈ 556 m at equator; should be in that range
        assert 400 < dist < 700

    def test_equatorial_point(self):
        p1 = GPSPoint(timestamp=None, latitude=0.0, longitude=0.0)
        p2 = GPSPoint(timestamp=None, latitude=0.0, longitude=1.0)
        dist = haversine_distance(p1, p2)
        # 1 degree longitude at equator ≈ 111_320 m
        assert 110_000 < dist < 113_000


# ---------------------------------------------------------------------------
# GPS CSV parser tests
# ---------------------------------------------------------------------------

class TestGPSCSVParser:
    def test_standard_columns(self):
        parser = GPSCSVParser()
        track = parser.parse(SAMPLE_CSV_STANDARD, "test.csv")
        assert track.available
        assert len(track.points) == 3
        assert track.source_mode == "csv"

    def test_alternate_columns(self):
        parser = GPSCSVParser()
        track = parser.parse(SAMPLE_CSV_ALTERNATE_COLS, "test.csv")
        assert track.available
        assert len(track.points) == 2

    def test_bad_csv(self):
        parser = GPSCSVParser()
        track = parser.parse(SAMPLE_CSV_BAD, "bad.csv")
        assert not track.available
        assert len(track.parse_warnings) > 0

    def test_invalid_coordinates_filtered(self):
        parser = GPSCSVParser()
        track = parser.parse(SAMPLE_CSV_INVALID_COORDS, "invalid.csv")
        # (0,0) is filtered as invalid; 91 degree lat is also invalid
        assert len(track.points) == 0

    def test_accuracy_parsed(self):
        parser = GPSCSVParser()
        track = parser.parse(SAMPLE_CSV_STANDARD, "test.csv")
        assert track.points[0].accuracy_m == pytest.approx(4.5)

    def test_timestamp_parsed(self):
        parser = GPSCSVParser()
        track = parser.parse(SAMPLE_CSV_STANDARD, "test.csv")
        ts = track.points[0].timestamp
        assert ts is not None
        assert ts.year == 2025


# ---------------------------------------------------------------------------
# GPS interpolation tests
# ---------------------------------------------------------------------------

class TestGPSInterpolation:
    def _make_track(self) -> GPSTrack:
        track = GPSTrack(source_mode="csv")
        track.points = [
            GPSPoint(
                timestamp=datetime(2025, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
                latitude=12.9716, longitude=77.5942, accuracy_m=5.0, source="csv"
            ),
            GPSPoint(
                timestamp=datetime(2025, 1, 15, 8, 0, 10, tzinfo=timezone.utc),
                latitude=12.9726, longitude=77.5952, accuracy_m=4.0, source="csv"
            ),
        ]
        return track

    def test_nearest_point(self):
        track = self._make_track()
        ts = datetime(2025, 1, 15, 8, 0, 2, tzinfo=timezone.utc)
        pt = track.nearest_point(ts)
        assert pt is not None
        assert pt.latitude == pytest.approx(12.9716)

    def test_interpolated_midpoint(self):
        track = self._make_track()
        ts = datetime(2025, 1, 15, 8, 0, 5, tzinfo=timezone.utc)
        pt = track.interpolated_point(ts)
        assert pt is not None
        # Midpoint between (12.9716, 77.5942) and (12.9726, 77.5952)
        assert pt.latitude == pytest.approx(12.9721, abs=0.0001)
        assert pt.longitude == pytest.approx(77.5947, abs=0.0001)

    def test_interpolated_before_start(self):
        track = self._make_track()
        ts = datetime(2025, 1, 15, 7, 59, 0, tzinfo=timezone.utc)
        pt = track.interpolated_point(ts)
        assert pt is not None  # Returns first point
        assert pt.latitude == pytest.approx(12.9716)

    def test_no_gps_empty_track(self):
        track = GPSTrack(source_mode="none")
        assert not track.available
        assert track.nearest_point(datetime.now(timezone.utc)) is None

    def test_total_distance(self):
        track = self._make_track()
        dist = track.total_distance_m
        assert dist > 0


# ---------------------------------------------------------------------------
# GPS Manager facade tests
# ---------------------------------------------------------------------------

class TestGPSManager:
    def test_from_csv_returns_track(self):
        mgr = GPSManager()
        track = mgr.from_csv(SAMPLE_CSV_STANDARD, "test.csv")
        assert track.available

    def test_empty_track(self):
        mgr = GPSManager()
        track = mgr.empty_track()
        assert not track.available
        assert track.source_mode == "none"

    def test_gps_point_valid(self):
        pt = GPSPoint(timestamp=None, latitude=12.97, longitude=77.59)
        assert pt.is_valid()

    def test_gps_point_invalid_null_island(self):
        pt = GPSPoint(timestamp=None, latitude=0.0, longitude=0.0)
        assert not pt.is_valid()

    def test_gps_point_accuracy_label(self):
        pt = GPSPoint(timestamp=None, latitude=12.97, longitude=77.59, accuracy_m=3.0)
        assert "high" in pt.accuracy_label()

        pt2 = GPSPoint(timestamp=None, latitude=12.97, longitude=77.59, accuracy_m=50.0)
        assert "low" in pt2.accuracy_label()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
