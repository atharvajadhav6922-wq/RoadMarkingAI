"""
src/inspection_pipeline.py

Orchestrates the complete inspection workflow:

  1. Accept uploaded files
  2. Extract/load GPS
  3. Process videos (frame sampling + detection + temporal tracking)
  4. Process photographs (single-frame detection)
  5. Run defect detector
  6. Query standards engine
  7. Save evidence
  8. Return InspectionResult

This module keeps app.py thin — all heavy logic lives here.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import PROCESSED_DIR
from src.defect_detector import DefectDetector, DefectRecord
from src.evidence_manager import EvidenceManager, EvidenceItem
from src.gps_manager import GPSManager, GPSTrack, GPSPoint
from src.image_processor import ImageProcessor, PhotoResult
from src.input_manager import InputSession, UploadedFile
from src.lane_detector import build_detector
from src.map_generator import generate_map
from src.road_classification import RoadParameters
from src.standards_engine import StandardsEngine, StandardsResult
from src.temporal_tracker import TemporalTracker
from src.video_processor import VideoProcessor, VideoMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class InspectionResult:
    session_id: str
    road_params: RoadParameters
    defects: List[DefectRecord] = field(default_factory=list)
    photo_results: List[PhotoResult] = field(default_factory=list)
    video_meta: List[VideoMetadata] = field(default_factory=list)
    gps_track: Optional[GPSTrack] = None
    standards_result: Optional[StandardsResult] = None
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    total_frames_analysed: int = 0
    source_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    map_object: Optional[Any] = None

    @property
    def defect_count(self) -> int:
        return len(self.defects)

    @property
    def gps_summary(self) -> Dict[str, Any]:
        if self.gps_track and self.gps_track.available:
            pts = self.gps_track.points
            accs = [p.accuracy_m for p in pts if p.accuracy_m is not None]
            avg_acc = f"±{sum(accs)/len(accs):.1f} m" if accs else "unknown"
            return {
                "available": True,
                "mode": self.gps_track.source_mode,
                "point_count": len(pts),
                "total_distance_m": self.gps_track.total_distance_m,
                "avg_accuracy": avg_acc,
                "source": pts[0].source if pts else "unknown",
            }
        return {"available": False, "mode": "none"}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class InspectionPipeline:
    """
    Main inspection orchestrator.

    Usage:
        pipeline = InspectionPipeline(road_params)
        result = pipeline.run(
            session=uploaded_session,
            gps_csv_bytes=...,
            video_start_datetime=...,
            sample_fps=5.0,
            progress_callback=...,
        )
    """

    def __init__(self, road_params: RoadParameters):
        self.road_params = road_params
        self.session_id = str(uuid.uuid4())[:12]
        self._gps_manager = GPSManager()
        self._standards_engine = StandardsEngine()
        self._detector = build_detector()
        self._evidence_mgr = EvidenceManager(self.session_id)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        session: InputSession,
        gps_csv_bytes: Optional[bytes] = None,
        gps_csv_filename: str = "gps.csv",
        video_start_datetime: Optional[datetime] = None,
        sample_fps: float = 5.0,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> InspectionResult:

        result = InspectionResult(
            session_id=self.session_id,
            road_params=self.road_params,
        )

        # ── 1. GPS ──────────────────────────────────────────────────────────
        if gps_csv_bytes:
            result.gps_track = self._gps_manager.from_csv(gps_csv_bytes, gps_csv_filename)
            if result.gps_track.parse_warnings:
                result.warnings.extend(result.gps_track.parse_warnings)
        else:
            result.gps_track = self._gps_manager.empty_track()

        # ── 2. Process photographs ──────────────────────────────────────────
        img_processor = ImageProcessor(self._detector)
        exif_gps_points: List[GPSPoint] = []

        for f in session.images:
            if progress_callback:
                progress_callback(0.0, f"Processing photo: {f.original_name}")
            try:
                photo_res = img_processor.process_file(f.saved_path)
                result.photo_results.append(photo_res)
                result.source_files.append(f.original_name)
                result.total_frames_analysed += 1

                if photo_res.gps_point:
                    exif_gps_points.append(photo_res.gps_point)

                # Save evidence
                if photo_res.annotated_frame is not None:
                    orig = cv2.imread(str(f.saved_path))
                    ev = self._evidence_mgr.save_photo_evidence(
                        f.original_name,
                        orig if orig is not None else photo_res.annotated_frame,
                        photo_res.annotated_frame,
                        photo_res.to_dict(),
                    )
                    result.evidence_items.append(ev)

                if photo_res.error:
                    result.warnings.append(f"{f.original_name}: {photo_res.error}")

            except Exception as exc:
                result.errors.append(f"Photo error ({f.original_name}): {exc}")
                logger.exception(f"Photo processing failed: {f.original_name}")

        # If no CSV GPS, build from EXIF
        if not result.gps_track.available and exif_gps_points:
            result.gps_track = self._gps_manager.build_exif_track(exif_gps_points)

        # ── 3. Process videos ───────────────────────────────────────────────
        all_frame_results: List[Dict[str, Any]] = []

        for f in session.videos:
            if progress_callback:
                progress_callback(0.0, f"Processing video: {f.original_name}")
            try:
                vp = VideoProcessor(
                    detector=self._detector,
                    sample_fps=sample_fps,
                )
                meta, frame_results = vp.process(
                    f.saved_path,
                    gps_track=result.gps_track,
                    video_start_datetime=video_start_datetime,
                    progress_callback=progress_callback,
                )
                result.video_meta.append(meta)
                result.source_files.append(f.original_name)
                result.total_frames_analysed += len(frame_results)

                if meta.error:
                    result.errors.append(f"{f.original_name}: {meta.error}")
                    continue

                all_frame_results.extend(frame_results)

            except Exception as exc:
                result.errors.append(f"Video error ({f.original_name}): {exc}")
                logger.exception(f"Video processing failed: {f.original_name}")

        # ── 4. Defect detection (video) ─────────────────────────────────────
        if all_frame_results:
            detector = DefectDetector(
                expected_marking=self.road_params.expected_marking,
                road_classification=self.road_params.road_class,
                divided=self.road_params.divided,
                num_lanes=self.road_params.num_lanes,
            )
            video_defects = detector.analyse(
                frame_results=all_frame_results,
                fps=sample_fps,
                source_file=", ".join(f.original_name for f in session.videos),
            )
            result.defects.extend(video_defects)

            # Save defect evidence
            frame_index_map = {fr["frame_index"]: fr for fr in all_frame_results}
            for d in video_defects:
                ev_fr = frame_index_map.get(d.evidence_frame_index, {})
                ann = ev_fr.get("annotated_frame")
                raw = ev_fr.get("raw_frame")
                if ann is not None:
                    ev_item = self._evidence_mgr.save_defect_evidence(d, raw, ann)
                    result.evidence_items.append(ev_item)

        # ── 5. Photo-level defects ──────────────────────────────────────────
        for photo_res in result.photo_results:
            if photo_res.detection and not photo_res.detection.detected:
                from src.defect_detector import DefectRecord, DefectType, DEFECT_RECOMMENDATIONS
                d = DefectRecord(
                    defect_type=DefectType.MISSING,
                    confidence=0.60,
                    start_time_s=0.0,
                    end_time_s=0.0,
                    latitude=photo_res.gps_point.latitude if photo_res.gps_point else None,
                    longitude=photo_res.gps_point.longitude if photo_res.gps_point else None,
                    gps_accuracy_m=photo_res.gps_point.accuracy_m if photo_res.gps_point else None,
                    gps_source=photo_res.gps_point.source if photo_res.gps_point else "none",
                    source_file=photo_res.path.name,
                    engineering_note="Single photo — absence may be due to framing. Field verification required.",
                )
                d.recommendation = DEFECT_RECOMMENDATIONS[DefectType.MISSING]
                result.defects.append(d)
            elif photo_res.detection and photo_res.detection.visibility_score < 0.45 and photo_res.detection.detected:
                from src.defect_detector import DefectRecord, DefectType, DEFECT_RECOMMENDATIONS
                d = DefectRecord(
                    defect_type=DefectType.FADED,
                    confidence=photo_res.detection.confidence,
                    start_time_s=0.0,
                    end_time_s=0.0,
                    latitude=photo_res.gps_point.latitude if photo_res.gps_point else None,
                    longitude=photo_res.gps_point.longitude if photo_res.gps_point else None,
                    gps_accuracy_m=photo_res.gps_point.accuracy_m if photo_res.gps_point else None,
                    gps_source=photo_res.gps_point.source if photo_res.gps_point else "none",
                    source_file=photo_res.path.name,
                )
                d.recommendation = DEFECT_RECOMMENDATIONS[DefectType.FADED]
                result.defects.append(d)

        # ── 6. Standards lookup ─────────────────────────────────────────────
        result.standards_result = self._standards_engine.query(
            road_class=self.road_params.road_class,
            marking_type=self.road_params.expected_marking,
            divided=self.road_params.divided,
        )

        # Attach standard references to defects
        for d in result.defects:
            if result.standards_result and result.standards_result.applicable_references:
                d.applicable_standard = result.standards_result.standards_list

        # ── 7. Map ──────────────────────────────────────────────────────────
        if result.gps_track.available:
            result.map_object = generate_map(result.defects)

        if progress_callback:
            progress_callback(1.0, "Inspection complete")

        return result
