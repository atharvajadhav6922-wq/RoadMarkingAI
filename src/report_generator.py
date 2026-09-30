"""
src/report_generator.py

Generates a professional PDF inspection report using ReportLab.

Sections:
  1. Executive Summary
  2. Inspection Details
  3. Road Characteristics
  4. Data Sources
  5. GPS Information
  6. Detected Defects
  7. Defect Location Table
  8. Photographic Evidence
  9. Applicable Standards
  10. Recommended Rectification
  11. Engineering Review Requirements
  12. Limitations / Disclaimer
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import KeepTogether

from config.settings import DISCLAIMER_TEXT, REPORT_VERSION, REPORTS_DIR
from src.defect_detector import DefectRecord, DefectType, DEFECT_RECOMMENDATIONS
from src.road_classification import RoadParameters
from src.standards_engine import StandardsResult

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOUR_HEADER = colors.HexColor("#1a3c5e")
COLOUR_ACCENT = colors.HexColor("#0074b3")
COLOUR_TABLE_HEADER = colors.HexColor("#2d6a9f")
COLOUR_ROW_ALT = colors.HexColor("#f0f6fb")
COLOUR_MISSING = colors.HexColor("#c0392b")
COLOUR_FADED = colors.HexColor("#e67e22")
COLOUR_BROKEN = colors.HexColor("#d35400")
COLOUR_INCORRECT = colors.HexColor("#8e44ad")
COLOUR_OK = colors.HexColor("#27ae60")
COLOUR_WARNING = colors.HexColor("#e74c3c")

DEFECT_COLOURS_RL = {
    DefectType.MISSING: COLOUR_MISSING,
    DefectType.FADED: COLOUR_FADED,
    DefectType.BROKEN: COLOUR_BROKEN,
    DefectType.INCORRECT_ARRANGEMENT: COLOUR_INCORRECT,
}


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Builds a ReportLab PDF for one inspection session."""

    def __init__(self):
        self._styles = self._build_styles()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate(
        self,
        road_params: RoadParameters,
        defects: List[DefectRecord],
        standards_result: Optional[StandardsResult],
        gps_summary: Dict[str, Any],
        session_id: str,
        source_files: List[str],
        total_frames: int = 0,
    ) -> Path:
        """
        Generate a PDF report and save it to REPORTS_DIR.
        Returns the path to the saved PDF.
        """
        filename = f"RoadMarkingAI_Report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        out_path = REPORTS_DIR / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        story = []
        story += self._section_cover(road_params, session_id)
        story += self._section_executive_summary(road_params, defects, total_frames)
        story += self._section_inspection_details(road_params, source_files)
        story += self._section_road_characteristics(road_params)
        story += self._section_gps(gps_summary)
        story += self._section_defects(defects)
        story += self._section_defect_table(defects)
        story += self._section_evidence_images(defects)
        story += self._section_standards(standards_result)
        story += self._section_rectification(defects)
        story += self._section_engineering_review(defects)
        story += self._section_limitations()

        doc = BaseDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN,
            bottomMargin=MARGIN,
            title="Road Marking Condition Assessment Report",
            author="RoadMarkingAI",
            subject=f"Inspection: {road_params.road_name or 'Unnamed Road'}",
        )
        frame = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN)
        template = PageTemplate(id="main", frames=frame, onPage=self._header_footer)
        doc.addPageTemplates([template])

        try:
            doc.build(story)
        except Exception as exc:
            logger.exception(f"PDF build error: {exc}")
            raise

        return out_path

    def generate_bytes(self, *args, **kwargs) -> bytes:
        """Generate PDF and return as bytes (for Streamlit download)."""
        path = self.generate(*args, **kwargs)
        return path.read_bytes()

    # -----------------------------------------------------------------------
    # Header / Footer
    # -----------------------------------------------------------------------

    def _header_footer(self, canvas, doc):
        canvas.saveState()
        # Header bar
        canvas.setFillColor(COLOUR_HEADER)
        canvas.rect(MARGIN, PAGE_H - MARGIN - 0.8 * cm, PAGE_W - 2 * MARGIN, 0.8 * cm, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(MARGIN + 4 * mm, PAGE_H - MARGIN - 0.55 * cm, "ROAD MARKING CONDITION ASSESSMENT REPORT")
        canvas.drawRightString(PAGE_W - MARGIN - 4 * mm, PAGE_H - MARGIN - 0.55 * cm, f"v{REPORT_VERSION}")

        # Footer
        canvas.setFillColor(colors.grey)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(MARGIN, MARGIN - 0.5 * cm, "CONFIDENTIAL — ENGINEERING INSPECTION ASSISTANCE ONLY")
        canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 0.5 * cm, f"Page {doc.page}")
        canvas.restoreState()

    # -----------------------------------------------------------------------
    # Section builders
    # -----------------------------------------------------------------------

    def _section_cover(self, rp: RoadParameters, session_id: str) -> list:
        s = self._styles
        story = [Spacer(1, 2 * cm)]
        story.append(Paragraph("ROAD MARKING CONDITION ASSESSMENT REPORT", s["cover_title"]))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Indian Urban Road Marking Inspection System", s["cover_subtitle"]))
        story.append(Spacer(1, 0.5 * cm))
        story.append(HRFlowable(width="100%", thickness=2, color=COLOUR_ACCENT))
        story.append(Spacer(1, 0.5 * cm))

        info = [
            ["Road Name", rp.road_name or "—"],
            ["Responsible Authority", rp.authority or "—"],
            ["Inspector", rp.inspector_name or "—"],
            ["Inspection Date", rp.inspection_date or "—"],
            ["Road Classification", rp.road_class_display],
            ["Road Geometry", rp.geometry_label],
            ["Report Generated", datetime.now().strftime("%d %b %Y  %H:%M")],
            ["Session ID", session_id],
            ["Report Version", REPORT_VERSION],
        ]
        table = Table(info, colWidths=[5.5 * cm, 11 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), COLOUR_ROW_ALT),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.5 * cm))

        # Disclaimer box on cover
        story.append(self._disclaimer_box())
        story.append(PageBreak())
        return story

    def _section_executive_summary(self, rp: RoadParameters, defects: List[DefectRecord], total_frames: int) -> list:
        s = self._styles
        story = [Paragraph("1. Executive Summary", s["h1"]), Spacer(1, 0.3 * cm)]

        counts = {t: 0 for t in DefectType}
        for d in defects:
            counts[d.defect_type] += 1

        confirmed = sum(1 for d in defects if "Confirmed" in d.review_status)
        pending = sum(1 for d in defects if "pending" in d.review_status)

        summary_text = (
            f"This AI-assisted inspection analysed road marking conditions on <b>{rp.road_name or 'the inspected road'}</b>. "
            f"A total of <b>{len(defects)}</b> potential marking defects were identified across <b>{total_frames}</b> analysed frames. "
            f"Of these, <b>{confirmed}</b> have been confirmed by the inspector and <b>{pending}</b> are pending review."
        )
        story.append(Paragraph(summary_text, s["body"]))
        story.append(Spacer(1, 0.3 * cm))

        # Summary table
        summary_data = [
            ["Defect Type", "Count", "Note"],
            ["Missing Markings", str(counts[DefectType.MISSING]), "Marking expected but absent over sustained segment"],
            ["Faded Markings", str(counts[DefectType.FADED]), "Marking present but low visibility"],
            ["Broken / Discontinuous", str(counts[DefectType.BROKEN]), "Unexplained gap in expected continuous marking"],
            ["Incorrect Arrangement", str(counts[DefectType.INCORRECT_ARRANGEMENT]), "Arrangement inconsistent with expected configuration"],
            ["TOTAL", str(len(defects)), "All AI-candidate defects"],
        ]
        t = Table(summary_data, colWidths=[5.5 * cm, 2.5 * cm, 8.5 * cm])
        t.setStyle(self._defect_table_style())
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            "<b>Note:</b> All results are AI candidates. Final determination must be made by a qualified engineer "
            "against the applicable IRC/MoRTH provisions and sanctioned drawings.", s["note"]
        ))
        story.append(PageBreak())
        return story

    def _section_inspection_details(self, rp: RoadParameters, source_files: List[str]) -> list:
        s = self._styles
        story = [Paragraph("2. Inspection Details", s["h1"]), Spacer(1, 0.3 * cm)]
        rows = [
            ["Parameter", "Value"],
            ["Road Name", rp.road_name or "—"],
            ["Responsible Authority", rp.authority or "—"],
            ["Inspector Name", rp.inspector_name or "—"],
            ["Inspection Date", rp.inspection_date or "—"],
            ["Inspection Direction", rp.inspection_direction or "—"],
            ["Expected Marking Config", rp.expected_marking_display],
            ["Source Files", "\n".join(source_files) if source_files else "—"],
        ]
        t = Table(rows, colWidths=[5.5 * cm, 11 * cm])
        t.setStyle(self._info_table_style())
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))
        return story

    def _section_road_characteristics(self, rp: RoadParameters) -> list:
        s = self._styles
        story = [Paragraph("3. Road Characteristics", s["h1"]), Spacer(1, 0.3 * cm)]
        rows = [
            ["Parameter", "Value"],
            ["Road Classification", rp.road_class_display],
            ["Divided / Undivided", "Divided" if rp.divided else "Undivided"],
            ["Number of Lanes", str(rp.num_lanes)],
            ["Carriageway Width", f"{rp.carriageway_width_m} m" if rp.carriageway_width_m else "—"],
            ["Lane Width", f"{rp.lane_width_m} m" if rp.lane_width_m else "—"],
            ["Shoulder Width", f"{rp.shoulder_width_m} m" if rp.shoulder_width_m else "—"],
            ["Pavement Type", rp.pavement_type.title()],
        ]
        t = Table(rows, colWidths=[5.5 * cm, 11 * cm])
        t.setStyle(self._info_table_style())
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))
        return story

    def _section_gps(self, gps_summary: Dict[str, Any]) -> list:
        s = self._styles
        story = [Paragraph("5. GPS Information", s["h1"]), Spacer(1, 0.3 * cm)]
        mode = gps_summary.get("mode", "none")
        available = gps_summary.get("available", False)

        if not available:
            story.append(Paragraph(
                "GPS data was not available for this inspection. Defect positions are reported as "
                "frame-based time offsets only. Real-world coordinates cannot be assigned.",
                s["body"],
            ))
        else:
            rows = [
                ["Parameter", "Value"],
                ["GPS Mode", mode.upper()],
                ["Points Recorded", str(gps_summary.get("point_count", 0))],
                ["Total Track Distance", f"{gps_summary.get('total_distance_m', 0):.1f} m"],
                ["Average Accuracy", gps_summary.get("avg_accuracy", "unknown")],
                ["GPS Source", gps_summary.get("source", "unknown")],
            ]
            t = Table(rows, colWidths=[5.5 * cm, 11 * cm])
            t.setStyle(self._info_table_style())
            story.append(t)
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(
                "<b>Accuracy note:</b> GPS accuracy reported here is the device-reported horizontal accuracy. "
                "Actual position uncertainty may be higher due to multipath, urban canyon effects, or satellite geometry. "
                "Do not assume better accuracy than the source GPS indicates.",
                s["note"],
            ))
        story.append(Spacer(1, 0.5 * cm))
        return story

    def _section_defects(self, defects: List[DefectRecord]) -> list:
        s = self._styles
        story = [Paragraph("6. Detected Defects (Detail)", s["h1"]), Spacer(1, 0.3 * cm)]

        if not defects:
            story.append(Paragraph("No potential defects were detected in this inspection.", s["body"]))
            story.append(Spacer(1, 0.5 * cm))
            return story

        for idx, d in enumerate(defects):
            color_rl = DEFECT_COLOURS_RL.get(d.defect_type, COLOUR_MISSING)
            block = [
                Paragraph(
                    f"<font color='#{self._rl_color_hex(color_rl)}'><b>[{d.defect_id}] {d.defect_type.value}</b></font>",
                    s["defect_title"]
                ),
            ]
            info_rows = [
                ["Defect ID", d.defect_id, "Review Status", d.review_status],
                ["Type", d.defect_type.value, "Confidence", f"{d.confidence:.2f}"],
                ["Start Time", f"{d.start_time_s:.1f} s", "End Time", f"{d.end_time_s:.1f} s"],
                ["Duration", f"{d.duration_s:.1f} s", "Road Position", d.estimated_road_position or "—"],
                ["Latitude", f"{d.latitude:.6f}" if d.latitude else "—",
                 "Longitude", f"{d.longitude:.6f}" if d.longitude else "—"],
                ["GPS Accuracy", d.gps_accuracy_m and f"±{d.gps_accuracy_m:.1f} m" or "—",
                 "GPS Source", d.gps_source],
                ["Source File", d.source_file, "Standard", d.applicable_standard or "—"],
            ]
            t = Table(info_rows, colWidths=[3.5 * cm, 6.0 * cm, 3.5 * cm, 3.5 * cm])
            t.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), COLOUR_ROW_ALT),
                ("BACKGROUND", (2, 0), (2, -1), COLOUR_ROW_ALT),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            block.append(t)
            block.append(Spacer(1, 0.15 * cm))
            block.append(Paragraph(f"<b>Recommendation:</b> {d.recommendation}", s["note"]))
            if d.engineering_note:
                block.append(Paragraph(f"<b>Engineering Note:</b> {d.engineering_note}", s["note"]))
            block.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
            block.append(Spacer(1, 0.2 * cm))
            story.append(KeepTogether(block))

        story.append(Spacer(1, 0.3 * cm))
        return story

    def _section_defect_table(self, defects: List[DefectRecord]) -> list:
        s = self._styles
        story = [Paragraph("7. Defect Location Table", s["h1"]), Spacer(1, 0.3 * cm)]

        if not defects:
            story.append(Paragraph("No defects detected.", s["body"]))
            story.append(Spacer(1, 0.3 * cm))
            return story

        header = ["ID", "Type", "Conf.", "Lat", "Lon", "Time (s)", "Review"]
        rows = [header]
        for d in defects:
            rows.append([
                d.defect_id,
                d.defect_type.value,
                f"{d.confidence:.2f}",
                f"{d.latitude:.5f}" if d.latitude else "—",
                f"{d.longitude:.5f}" if d.longitude else "—",
                f"{d.start_time_s:.1f}–{d.end_time_s:.1f}",
                d.review_status[:20],
            ])

        col_widths = [1.5 * cm, 3.5 * cm, 1.5 * cm, 2.8 * cm, 2.8 * cm, 2.5 * cm, 2.0 * cm]
        t = Table(rows, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOUR_TABLE_HEADER),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOUR_ROW_ALT]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ALIGN", (2, 1), (4, -1), "CENTER"),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))
        story.append(PageBreak())
        return story

    def _section_evidence_images(self, defects: List[DefectRecord]) -> list:
        s = self._styles
        story = [Paragraph("8. Photographic Evidence", s["h1"]), Spacer(1, 0.3 * cm)]

        evidence_added = 0
        for d in defects:
            # Look for annotated image
            ann_path = self._find_evidence_image(d)
            if ann_path and ann_path.exists():
                try:
                    img = Image(str(ann_path), width=14 * cm, height=8 * cm, kind="proportional")
                    block = [
                        Paragraph(f"<b>[{d.defect_id}] {d.defect_type.value}</b> — Confidence: {d.confidence:.2f}  |  {d.review_status}", s["caption"]),
                        img,
                        Spacer(1, 0.3 * cm),
                    ]
                    story.append(KeepTogether(block))
                    evidence_added += 1
                except Exception as exc:
                    logger.warning(f"Cannot embed image {ann_path}: {exc}")

        if evidence_added == 0:
            story.append(Paragraph(
                "No evidence images were available. Ensure the application was run in processing mode "
                "and evidence was saved.", s["body"]
            ))

        story.append(Spacer(1, 0.3 * cm))
        story.append(PageBreak())
        return story

    def _section_standards(self, sr: Optional[StandardsResult]) -> list:
        s = self._styles
        story = [Paragraph("9. Applicable Standards", s["h1"]), Spacer(1, 0.3 * cm)]

        if sr is None or not sr.applicable_references:
            story.append(Paragraph(
                "No specific standard rules were matched for this road classification. "
                "Verify against IRC:35-2015 and MoRTH Section 803 (latest editions) and the sanctioned layout.",
                s["body"],
            ))
        else:
            for ref in sr.applicable_references:
                story.append(Paragraph(f"<b>{ref.get('id', '')} — {ref.get('standard', '')}</b>", s["h3"]))
                story.append(Paragraph(f"<b>Section:</b> {ref.get('section', '—')}", s["body"]))
                story.append(Paragraph(f"<b>Applicability:</b> {ref.get('applicability_notes', '—')}", s["body"]))
                story.append(Paragraph(f"<b>Engineering Note:</b> {ref.get('engineering_note', '—')}", s["note"]))
                story.append(Paragraph(f"<b>Source:</b> {ref.get('source_document', '—')}", s["body"]))
                story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(sr.disclaimer, s["note"]))

        story.append(Spacer(1, 0.3 * cm))
        return story

    def _section_rectification(self, defects: List[DefectRecord]) -> list:
        s = self._styles
        story = [Paragraph("10. Recommended Rectification", s["h1"]), Spacer(1, 0.3 * cm)]
        story.append(Paragraph(
            "The following recommendations are engineering guidance only. They do not constitute a legal notice or "
            "specification. All rectification shall be carried out in accordance with the applicable IRC/MoRTH "
            "provisions, sanctioned drawings, and approved project specification.",
            s["note"],
        ))
        story.append(Spacer(1, 0.2 * cm))

        for dtype in DefectType:
            typed_defects = [d for d in defects if d.defect_type == dtype]
            if typed_defects:
                story.append(Paragraph(f"<b>{dtype.value} ({len(typed_defects)} instance(s))</b>", s["h3"]))
                story.append(Paragraph(DEFECT_RECOMMENDATIONS[dtype], s["body"]))
                story.append(Spacer(1, 0.2 * cm))

        story.append(Spacer(1, 0.3 * cm))
        return story

    def _section_engineering_review(self, defects: List[DefectRecord]) -> list:
        s = self._styles
        story = [Paragraph("11. Engineering Review Requirements", s["h1"]), Spacer(1, 0.3 * cm)]
        story.append(Paragraph(
            "Each AI-candidate defect must be reviewed by a qualified engineer before any rectification order "
            "is issued. The review status is shown in the table above.",
            s["body"],
        ))
        story.append(Spacer(1, 0.2 * cm))

        pending = [d for d in defects if "pending" in d.review_status]
        confirmed = [d for d in defects if "Confirmed" in d.review_status]
        rejected = [d for d in defects if "Rejected" in d.review_status or "reject" in d.review_status.lower()]

        rows = [
            ["Status", "Count"],
            ["AI Candidate — Pending Review", str(len(pending))],
            ["Confirmed by Inspector", str(len(confirmed))],
            ["Rejected by Inspector", str(len(rejected))],
            ["Total", str(len(defects))],
        ]
        t = Table(rows, colWidths=[9 * cm, 7.5 * cm])
        t.setStyle(self._info_table_style())
        story.append(t)
        story.append(Spacer(1, 0.3 * cm))
        return story

    def _section_limitations(self) -> list:
        s = self._styles
        story = [Paragraph("12. Limitations", s["h1"]), Spacer(1, 0.3 * cm)]
        limitations = [
            "Detection is based on classical computer-vision algorithms (HSV thresholding + Hough Transform). "
            "A trained segmentation model is not included in the baseline and must be added separately.",
            "Retroreflectivity of markings cannot be determined from RGB imagery. Field measurement with a "
            "calibrated retroreflectometer is required for retroreflectivity assessment.",
            "Detection accuracy may be significantly reduced under: wet/reflective pavement, severe glare, "
            "night conditions, heavy traffic occlusion, dust, and paint/patch similar in colour to markings.",
            "GPS accuracy is limited by the source device. Urban canyon effects can cause lateral errors "
            "of several metres. Position accuracy of ±1–5 m is achievable with modern smartphone GPS under "
            "clear sky conditions but cannot be guaranteed.",
            "Distance estimation without GPS is based on approximate perspective geometry and is labelled "
            "'estimated' throughout. It does not correspond to a calibrated measurement.",
            "This report does not assess marking retroreflectivity, paint thickness, material type, or "
            "durability — all of which are required for full IRC/MoRTH compliance assessment.",
            "The standards database contains reference information only. All requirements must be verified "
            "against the current official IRC/MoRTH editions and project-specific sanction.",
        ]
        for i, lim in enumerate(limitations):
            story.append(Paragraph(f"{i+1}. {lim}", s["body"]))
            story.append(Spacer(1, 0.1 * cm))

        story.append(Spacer(1, 0.4 * cm))
        story.append(self._disclaimer_box())
        return story

    # -----------------------------------------------------------------------
    # Style helpers
    # -----------------------------------------------------------------------

    def _build_styles(self) -> dict:
        base = getSampleStyleSheet()
        styles = {}

        styles["cover_title"] = ParagraphStyle(
            "cover_title", parent=base["Title"],
            fontSize=22, textColor=COLOUR_HEADER, alignment=TA_CENTER, spaceAfter=10,
        )
        styles["cover_subtitle"] = ParagraphStyle(
            "cover_subtitle", parent=base["Normal"],
            fontSize=13, textColor=COLOUR_ACCENT, alignment=TA_CENTER, spaceAfter=6,
        )
        styles["h1"] = ParagraphStyle(
            "h1", parent=base["Heading1"],
            fontSize=13, textColor=COLOUR_HEADER, spaceAfter=6,
        )
        styles["h2"] = ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontSize=11, textColor=COLOUR_ACCENT, spaceAfter=4,
        )
        styles["h3"] = ParagraphStyle(
            "h3", parent=base["Heading3"],
            fontSize=10, textColor=COLOUR_HEADER, spaceAfter=2,
        )
        styles["body"] = ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=9, alignment=TA_JUSTIFY, spaceAfter=4, leading=14,
        )
        styles["note"] = ParagraphStyle(
            "note", parent=base["Normal"],
            fontSize=8, textColor=colors.HexColor("#555555"),
            alignment=TA_JUSTIFY, spaceAfter=4, leading=12,
            leftIndent=10,
        )
        styles["caption"] = ParagraphStyle(
            "caption", parent=base["Normal"],
            fontSize=8, textColor=COLOUR_ACCENT, alignment=TA_CENTER, spaceAfter=2,
        )
        styles["defect_title"] = ParagraphStyle(
            "defect_title", parent=base["Normal"],
            fontSize=11, spaceAfter=4, leading=14,
        )
        styles["disclaimer"] = ParagraphStyle(
            "disclaimer", parent=base["Normal"],
            fontSize=8, textColor=colors.white, alignment=TA_JUSTIFY, leading=12,
        )
        return styles

    def _disclaimer_box(self):
        from reportlab.platypus import Table, TableStyle
        text = Paragraph(DISCLAIMER_TEXT.replace("\n", "<br/>"), self._styles["disclaimer"])
        t = Table([[text]], colWidths=[PAGE_W - 2 * MARGIN - 1 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOUR_WARNING),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        return t

    @staticmethod
    def _info_table_style():
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOUR_TABLE_HEADER),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOUR_ROW_ALT]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])

    @staticmethod
    def _defect_table_style():
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOUR_TABLE_HEADER),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOUR_ROW_ALT]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#eaf0f6")),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ])

    @staticmethod
    def _rl_color_hex(c) -> str:
        """Convert ReportLab HexColor to plain hex string for XML embedding."""
        try:
            hex_val = c.hexval()
            return hex_val.lstrip("#")
        except Exception:
            return "c0392b"

    @staticmethod
    def _find_evidence_image(defect: DefectRecord) -> Optional[Path]:
        from config.settings import PROCESSED_DIR
        # Pattern: data/processed/<any_session>/evidence/<type>_<id>_annotated.jpg
        for session_dir in PROCESSED_DIR.iterdir():
            if not session_dir.is_dir():
                continue
            ev_dir = session_dir / "evidence"
            if not ev_dir.exists():
                continue
            for f in ev_dir.glob(f"*{defect.defect_id}_annotated.jpg"):
                return f
        return None
