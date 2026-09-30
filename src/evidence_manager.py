"""
src/evidence_manager.py

Saves annotated evidence images and manages the evidence directory.

Each defect gets:
  - original frame saved as JPEG
  - annotated frame saved as JPEG
  - metadata stored alongside

Evidence is stored in data/processed/<session_id>/evidence/
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config.settings import PROCESSED_DIR
from src.defect_detector import DefectRecord, DefectType

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    defect_id: str
    defect_type: str
    original_path: Optional[Path] = None
    annotated_path: Optional[Path] = None
    metadata_path: Optional[Path] = None


class EvidenceManager:
    """Manages evidence files for an inspection session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.evidence_dir = PROCESSED_DIR / session_id / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def save_defect_evidence(
        self,
        defect: DefectRecord,
        original_frame: Optional[np.ndarray],
        annotated_frame: Optional[np.ndarray],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> EvidenceItem:
        """
        Save original + annotated frames for a defect.
        Returns paths to saved files.
        """
        item = EvidenceItem(
            defect_id=defect.defect_id,
            defect_type=defect.defect_type.value,
        )

        prefix = f"{defect.defect_type.value}_{defect.defect_id}"

        if original_frame is not None:
            orig_path = self.evidence_dir / f"{prefix}_original.jpg"
            try:
                cv2.imwrite(str(orig_path), original_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                item.original_path = orig_path
            except Exception as exc:
                logger.warning(f"Could not save original frame: {exc}")

        if annotated_frame is not None:
            ann_path = self.evidence_dir / f"{prefix}_annotated.jpg"
            # Draw defect overlay on annotated frame
            marked = self._draw_defect_overlay(annotated_frame, defect)
            try:
                cv2.imwrite(str(ann_path), marked, [cv2.IMWRITE_JPEG_QUALITY, 90])
                item.annotated_path = ann_path
            except Exception as exc:
                logger.warning(f"Could not save annotated frame: {exc}")

        # Save metadata JSON
        meta = defect.to_dict()
        if extra_meta:
            meta.update(extra_meta)
        meta_path = self.evidence_dir / f"{prefix}_meta.json"
        try:
            meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            item.metadata_path = meta_path
        except Exception as exc:
            logger.warning(f"Could not save metadata: {exc}")

        return item

    def save_photo_evidence(
        self,
        filename: str,
        original_frame: np.ndarray,
        annotated_frame: np.ndarray,
        meta: Dict[str, Any],
    ) -> EvidenceItem:
        """Save evidence for a photograph-based detection."""
        stem = Path(filename).stem
        item = EvidenceItem(defect_id=stem, defect_type="photo")

        orig_path = self.evidence_dir / f"photo_{stem}_original.jpg"
        ann_path = self.evidence_dir / f"photo_{stem}_annotated.jpg"
        meta_path = self.evidence_dir / f"photo_{stem}_meta.json"

        try:
            cv2.imwrite(str(orig_path), original_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            item.original_path = orig_path
        except Exception:
            pass
        try:
            cv2.imwrite(str(ann_path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            item.annotated_path = ann_path
        except Exception:
            pass
        try:
            meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            item.metadata_path = meta_path
        except Exception:
            pass

        return item

    @staticmethod
    def _draw_defect_overlay(frame: np.ndarray, defect: DefectRecord) -> np.ndarray:
        """Draw defect information overlay on the annotated frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Colour coding
        colour_map = {
            DefectType.MISSING: (0, 0, 255),            # red
            DefectType.FADED: (0, 165, 255),             # orange
            DefectType.BROKEN: (0, 140, 255),            # dark orange
            DefectType.INCORRECT_ARRANGEMENT: (128, 0, 128),  # purple
        }
        color = colour_map.get(defect.defect_type, (0, 0, 255))

        # Bounding box if available
        if defect.pixel_bbox:
            x1, y1, x2, y2 = defect.pixel_bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        # Banner
        banner_lines = [
            f"DEFECT: {defect.defect_type.value}  ID:{defect.defect_id}",
            f"Conf: {defect.confidence:.2f}  {defect.review_status}",
        ]
        if defect.latitude:
            banner_lines.append(f"GPS: {defect.latitude:.6f},{defect.longitude:.6f}")
        if defect.estimated_road_position:
            banner_lines.append(f"Position: {defect.estimated_road_position}")

        for i, line in enumerate(banner_lines):
            y = 30 + i * 28
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return out

    def list_evidence(self) -> List[EvidenceItem]:
        """List all evidence items in the session directory."""
        items = []
        for meta_file in self.evidence_dir.glob("*_meta.json"):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                item = EvidenceItem(
                    defect_id=data.get("ID", meta_file.stem),
                    defect_type=data.get("Type", "unknown"),
                )
                prefix = meta_file.stem.replace("_meta", "")
                orig = self.evidence_dir / f"{prefix}_original.jpg"
                ann = self.evidence_dir / f"{prefix}_annotated.jpg"
                item.original_path = orig if orig.exists() else None
                item.annotated_path = ann if ann.exists() else None
                item.metadata_path = meta_file
                items.append(item)
            except Exception:
                pass
        return items
