"""
src/gps_manager.py

Handles all GPS data sources:
  MODE 1: Phone GPS CSV upload
  MODE 2: Photo EXIF GPS
  MODE 3: No GPS (graceful degradation)

GPS is associated with video frames using nearest-timestamp or linear
interpolation. Distance is calculated via the Haversine formula.

Important: we never claim accuracy better than the source GPS indicates.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.settings import (
    EARTH_RADIUS_M,
    GPS_CSV_POSSIBLE_ACC_COLS,
    GPS_CSV_POSSIBLE_LAT_COLS,
    GPS_CSV_POSSIBLE_LON_COLS,
    GPS_CSV_POSSIBLE_TIMESTAMP_COLS,
    GPS_MAX_INTERPOLATION_GAP_S,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class GPSPoint:
    timestamp: Optional[datetime]
    latitude: float
    longitude: float
    accuracy_m: Optional[float] = None   # horizontal accuracy in metres
    source: str = "unknown"              # "csv", "exif", "interpolated"

    def is_valid(self) -> bool:
        return (
            -90.0 <= self.latitude <= 90.0
            and -180.0 <= self.longitude <= 180.0
            and not (self.latitude == 0.0 and self.longitude == 0.0)
        )

    def accuracy_label(self) -> str:
        if self.accuracy_m is None:
            return "unknown"
        if self.accuracy_m <= 5:
            return f"±{self.accuracy_m:.1f} m (high)"
        if self.accuracy_m <= 15:
            return f"±{self.accuracy_m:.1f} m (moderate)"
        return f"±{self.accuracy_m:.1f} m (low)"


@dataclass
class GPSTrack:
    points: List[GPSPoint] = field(default_factory=list)
    source_mode: str = "none"   # "csv", "exif", "none"
    parse_warnings: List[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return len(self.points) > 0

    @property
    def total_distance_m(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return sum(
            haversine_distance(self.points[i], self.points[i + 1])
            for i in range(len(self.points) - 1)
        )

    def nearest_point(self, ts: datetime) -> Optional[GPSPoint]:
        """Return the GPS point whose timestamp is nearest to ts."""
        if not self.points or ts is None:
            return None
        valid = [p for p in self.points if p.timestamp is not None]
        if not valid:
            return None
        return min(valid, key=lambda p: abs((p.timestamp - ts).total_seconds()))

    def interpolated_point(self, ts: datetime) -> Optional[GPSPoint]:
        """
        Linearly interpolate between two GPS points bracketing ts.
        If the gap is > GPS_MAX_INTERPOLATION_GAP_S, falls back to nearest.
        """
        if not self.points or ts is None:
            return None
        valid = sorted(
            [p for p in self.points if p.timestamp is not None],
            key=lambda p: p.timestamp,
        )
        if not valid:
            return None

        # Find bracketing points
        before = [p for p in valid if p.timestamp <= ts]
        after = [p for p in valid if p.timestamp >= ts]

        if not before:
            return after[0] if after else None
        if not after:
            return before[-1]

        p0 = before[-1]
        p1 = after[0]

        gap = (p1.timestamp - p0.timestamp).total_seconds()
        if gap <= 0:
            return p0
        if gap > GPS_MAX_INTERPOLATION_GAP_S:
            # Gap too large — return nearest with a warning flag
            nearest = self.nearest_point(ts)
            if nearest:
                # Clone and tag as interpolated anyway
                return GPSPoint(
                    timestamp=ts,
                    latitude=nearest.latitude,
                    longitude=nearest.longitude,
                    accuracy_m=nearest.accuracy_m,
                    source="nearest (gap warned)",
                )
            return None

        frac = (ts - p0.timestamp).total_seconds() / gap
        lat = p0.latitude + frac * (p1.latitude - p0.latitude)
        lon = p0.longitude + frac * (p1.longitude - p0.longitude)
        acc = None
        if p0.accuracy_m is not None and p1.accuracy_m is not None:
            acc = p0.accuracy_m + frac * (p1.accuracy_m - p0.accuracy_m)

        return GPSPoint(
            timestamp=ts,
            latitude=lat,
            longitude=lon,
            accuracy_m=acc,
            source="interpolated",
        )


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_distance(p1: GPSPoint, p2: GPSPoint) -> float:
    """Return great-circle distance between two GPS points in metres."""
    lat1, lon1 = math.radians(p1.latitude), math.radians(p1.longitude)
    lat2, lon2 = math.radians(p2.latitude), math.radians(p2.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_M * c


# ---------------------------------------------------------------------------
# GPS CSV Parser
# ---------------------------------------------------------------------------

class GPSCSVParser:
    """
    Parse a phone GPS CSV file.
    Attempts to auto-detect column names using fuzzy matching.
    """

    def parse(self, file_bytes: bytes, filename: str = "gps.csv") -> GPSTrack:
        track = GPSTrack(source_mode="csv")
        try:
            text = file_bytes.decode("utf-8-sig")
            df = pd.read_csv(io.StringIO(text))
        except Exception as exc:
            track.parse_warnings.append(f"CSV parse error: {exc}")
            return track

        df.columns = [c.strip().lower() for c in df.columns]

        ts_col = self._find_col(df, GPS_CSV_POSSIBLE_TIMESTAMP_COLS)
        lat_col = self._find_col(df, GPS_CSV_POSSIBLE_LAT_COLS)
        lon_col = self._find_col(df, GPS_CSV_POSSIBLE_LON_COLS)
        acc_col = self._find_col(df, GPS_CSV_POSSIBLE_ACC_COLS)

        if lat_col is None or lon_col is None:
            track.parse_warnings.append(
                "Could not find latitude/longitude columns in CSV. "
                f"Columns found: {list(df.columns)}"
            )
            return track

        for _, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            except (ValueError, KeyError):
                continue

            ts: Optional[datetime] = None
            if ts_col:
                try:
                    ts_raw = str(row[ts_col])
                    ts = pd.to_datetime(ts_raw, utc=True).to_pydatetime()
                except Exception:
                    pass

            acc: Optional[float] = None
            if acc_col:
                try:
                    acc = float(row[acc_col])
                except Exception:
                    pass

            pt = GPSPoint(
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                accuracy_m=acc,
                source="csv",
            )
            if pt.is_valid():
                track.points.append(pt)
            else:
                track.parse_warnings.append(f"Skipped invalid GPS row: lat={lat}, lon={lon}")

        if not track.points:
            track.parse_warnings.append("No valid GPS points found in CSV.")
        return track

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None


# ---------------------------------------------------------------------------
# EXIF GPS Extractor
# ---------------------------------------------------------------------------

class EXIFGPSExtractor:
    """Extract GPS from JPEG/PNG EXIF data using Pillow."""

    def extract(self, image_bytes: bytes) -> Optional[GPSPoint]:
        try:
            from PIL import Image
            from PIL.ExifTags import GPSTAGS, TAGS
            import io as _io

            img = Image.open(_io.BytesIO(image_bytes))
            exif_data = img._getexif()
            if not exif_data:
                return None

            gps_info: Dict = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    for gps_tag_id, gps_val in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_info[gps_tag] = gps_val

            if not gps_info:
                return None

            lat = self._dms_to_decimal(
                gps_info.get("GPSLatitude"),
                gps_info.get("GPSLatitudeRef", "N"),
            )
            lon = self._dms_to_decimal(
                gps_info.get("GPSLongitude"),
                gps_info.get("GPSLongitudeRef", "E"),
            )

            if lat is None or lon is None:
                return None

            # Timestamp from EXIF GPS
            ts: Optional[datetime] = None
            try:
                gps_date = gps_info.get("GPSDateStamp", "")
                gps_time = gps_info.get("GPSTimeStamp")
                if gps_date and gps_time:
                    h = int(float(gps_time[0]))
                    m = int(float(gps_time[1]))
                    s = float(gps_time[2])
                    date_parts = gps_date.split(":")
                    ts = datetime(
                        int(date_parts[0]), int(date_parts[1]), int(date_parts[2]),
                        h, m, int(s), tzinfo=timezone.utc
                    )
            except Exception:
                pass

            pt = GPSPoint(
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                accuracy_m=None,  # EXIF does not include accuracy
                source="exif",
            )
            return pt if pt.is_valid() else None

        except Exception:
            return None

    @staticmethod
    def _dms_to_decimal(dms, ref) -> Optional[float]:
        if dms is None:
            return None
        try:
            d = float(dms[0])
            m = float(dms[1])
            s = float(dms[2])
            decimal = d + m / 60.0 + s / 3600.0
            if ref in ("S", "W"):
                decimal = -decimal
            return decimal
        except Exception:
            return None


# ---------------------------------------------------------------------------
# GPS Manager — façade
# ---------------------------------------------------------------------------

class GPSManager:
    """
    Façade that unifies all GPS modes.
    """

    def __init__(self):
        self._csv_parser = GPSCSVParser()
        self._exif_extractor = EXIFGPSExtractor()

    def from_csv(self, file_bytes: bytes, filename: str = "gps.csv") -> GPSTrack:
        return self._csv_parser.parse(file_bytes, filename)

    def from_exif(self, image_bytes: bytes) -> Optional[GPSPoint]:
        return self._exif_extractor.extract(image_bytes)

    def build_exif_track(self, exif_points: List[GPSPoint]) -> GPSTrack:
        """Build a GPSTrack from a list of EXIF-sourced points (one per photo)."""
        track = GPSTrack(source_mode="exif")
        track.points = [p for p in exif_points if p is not None and p.is_valid()]
        return track

    def empty_track(self) -> GPSTrack:
        return GPSTrack(source_mode="none")
