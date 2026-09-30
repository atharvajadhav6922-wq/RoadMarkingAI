"""
app.py — RoadMarkingAI
Streamlit dashboard for Indian urban road marking inspection.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DISCLAIMER_TEXT, REPORTS_DIR, SUPPORTED_IMAGE_EXTS, SUPPORTED_VIDEO_EXTS
from src.defect_detector import DefectType
from src.input_manager import InputManager, classify_extension
from src.inspection_pipeline import InspectionPipeline, InspectionResult
from src.road_classification import (
    EXPECTED_MARKINGS,
    PAVEMENT_TYPES,
    ROAD_CLASSES,
    RoadParameters,
)
from src.report_generator import ReportGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RoadMarkingAI")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RoadMarkingAI — Indian Road Inspection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1a3c5e 0%, #0074b3 100%);
    color: white;
    padding: 2rem 2rem 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,116,179,0.25);
}
.main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
.main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 1rem; }

.metric-card {
    background: #ffffff;
    border: 1px solid #e0eaf3;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.15s ease;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .metric-value { font-size: 2.2rem; font-weight: 700; color: #1a3c5e; }
.metric-card .metric-label { font-size: 0.82rem; color: #666; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.05em; }

.defect-badge-MISSING     { background:#c0392b; color:white; padding:2px 8px; border-radius:4px; font-size:0.78rem; font-weight:600; }
.defect-badge-FADED       { background:#e67e22; color:white; padding:2px 8px; border-radius:4px; font-size:0.78rem; font-weight:600; }
.defect-badge-BROKEN      { background:#d35400; color:white; padding:2px 8px; border-radius:4px; font-size:0.78rem; font-weight:600; }
.defect-badge-INCORRECT_ARRANGEMENT { background:#8e44ad; color:white; padding:2px 8px; border-radius:4px; font-size:0.78rem; font-weight:600; }

.disclaimer-box {
    background: #c0392b;
    color: white;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    line-height: 1.6;
    margin-top: 1rem;
}

.gps-badge-available   { color: #27ae60; font-weight: 600; }
.gps-badge-unavailable { color: #e74c3c; font-weight: 600; }

.section-divider {
    border: none;
    border-top: 2px solid #e0eaf3;
    margin: 1.5rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ===========================================================================
# Session state helpers
# ===========================================================================

def init_session_state():
    defaults = {
        "session_id": str(uuid.uuid4())[:12],
        "inspection_result": None,
        "processing_done": False,
        "review_statuses": {},    # defect_id → status
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ===========================================================================
# Sidebar — Road Parameters
# ===========================================================================

def render_sidebar() -> Dict[str, Any]:
    st.sidebar.markdown("## 🛣️ Project / Road Information")

    road_name = st.sidebar.text_input("Road Name", placeholder="e.g. MG Road, Bengaluru")
    authority = st.sidebar.text_input("Responsible Authority", placeholder="e.g. BBMP / PWD / NHAI")
    inspection_date = st.sidebar.date_input("Inspection Date", value=date.today())
    inspector_name = st.sidebar.text_input("Inspector Name", placeholder="Full name")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Road Classification")
    road_class_display = st.sidebar.selectbox("Road Class", list(ROAD_CLASSES.keys()), index=0)

    st.sidebar.markdown("### Road Geometry")
    divided = st.sidebar.radio("Carriageway", ["Undivided", "Divided"], index=0) == "Divided"
    num_lanes = st.sidebar.number_input("Number of Lanes (total)", min_value=1, max_value=12, value=2, step=1)
    carriageway_width = st.sidebar.number_input("Carriageway Width (m)", min_value=0.0, value=0.0, step=0.5)
    lane_width = st.sidebar.number_input("Lane Width (m)", min_value=0.0, value=0.0, step=0.25)
    shoulder_width = st.sidebar.number_input("Shoulder Width (m)", min_value=0.0, value=0.0, step=0.25)

    pavement_display = st.sidebar.selectbox("Pavement Type", list(PAVEMENT_TYPES.keys()), index=0)
    inspection_direction = st.sidebar.text_input("Inspection Direction", placeholder="e.g. North-bound")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Expected Road Marking")
    expected_marking_display = st.sidebar.selectbox(
        "Expected Marking Configuration", list(EXPECTED_MARKINGS.keys()), index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Video Processing")
    sample_fps = st.sidebar.slider("Frame Sampling Rate (FPS)", min_value=1.0, max_value=15.0, value=5.0, step=1.0)

    return {
        "road_name": road_name,
        "authority": authority,
        "inspection_date": str(inspection_date),
        "inspector_name": inspector_name,
        "road_class_display": road_class_display,
        "divided": divided,
        "num_lanes": int(num_lanes),
        "carriageway_width_m": carriageway_width if carriageway_width > 0 else None,
        "lane_width_m": lane_width if lane_width > 0 else None,
        "shoulder_width_m": shoulder_width if shoulder_width > 0 else None,
        "pavement_type_display": pavement_display,
        "inspection_direction": inspection_direction,
        "expected_marking_display": expected_marking_display,
        "sample_fps": sample_fps,
    }


# ===========================================================================
# Header
# ===========================================================================

def render_header():
    st.markdown(
        """
<div class="main-header">
    <h1>🛣️ ROAD MARKING AI</h1>
    <p>Indian Urban Road Marking Inspection System &nbsp;|&nbsp; AI-Assisted Field Support Tool</p>
</div>
""",
        unsafe_allow_html=True,
    )


# ===========================================================================
# Input section
# ===========================================================================

def render_input_section() -> tuple:
    """Returns (uploaded_media_files, gps_csv_file, video_start_dt)"""
    st.markdown("## 📁 Upload Inspection Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        media_files = st.file_uploader(
            "Upload Photos and/or Videos",
            type=[e.lstrip(".") for e in (SUPPORTED_IMAGE_EXTS | SUPPORTED_VIDEO_EXTS)],
            accept_multiple_files=True,
            help="Upload one or more road photographs (JPG/PNG/etc.) or vehicle/CCTV videos (MP4/MOV/AVI/etc.)",
        )

    with col2:
        st.markdown("### GPS Data (optional)")
        gps_file = st.file_uploader(
            "Phone GPS CSV",
            type=["csv"],
            help="CSV with columns: timestamp, latitude, longitude, accuracy (or similar)",
        )
        video_start_str = st.text_input(
            "Video Start DateTime (UTC)",
            placeholder="2025-01-15 08:30:00",
            help="If GPS CSV is uploaded with a video, enter the video recording start time for GPS sync",
        )

    video_start_dt = None
    if video_start_str:
        try:
            from dateutil import parser as dateutil_parser
            video_start_dt = dateutil_parser.parse(video_start_str)
            if video_start_dt.tzinfo is None:
                from datetime import timezone
                video_start_dt = video_start_dt.replace(tzinfo=timezone.utc)
        except Exception:
            st.warning(f"Could not parse video start datetime: '{video_start_str}'. GPS sync disabled.")

    return media_files or [], gps_file, video_start_dt


def render_file_preview(uploaded_files: list):
    if not uploaded_files:
        return

    st.markdown("### Uploaded Files")
    cols = st.columns(min(len(uploaded_files), 3))

    for idx, uf in enumerate(uploaded_files):
        col = cols[idx % len(cols)]
        ftype = classify_extension(uf.name)
        size_mb = uf.size / (1024 * 1024)

        with col:
            if ftype == "image":
                try:
                    st.image(uf, caption=f"{uf.name}\n({size_mb:.2f} MB)", use_column_width=True)
                except Exception:
                    st.info(f"📷 {uf.name} ({size_mb:.2f} MB)")
            elif ftype == "video":
                st.video(uf)
                st.caption(f"🎥 {uf.name} ({size_mb:.2f} MB)")
            else:
                st.warning(f"⚠️ Unsupported: {uf.name}")

        uf.seek(0)  # Reset for downstream reading


# ===========================================================================
# Processing
# ===========================================================================

def run_inspection(
    media_files: list,
    gps_file,
    video_start_dt,
    road_params: RoadParameters,
    sample_fps: float,
) -> Optional[InspectionResult]:

    if not media_files:
        st.warning("Please upload at least one photo or video before processing.")
        return None

    # Save uploads
    with st.spinner("Saving uploaded files…"):
        mgr = InputManager(st.session_state["session_id"])
        session = mgr.process_uploads(media_files)

    # Errors
    errors = [f for f in session.files if f.file_type == "unsupported"]
    if errors:
        for e in errors:
            st.error(f"Unsupported file: {e.original_name}")

    if not session.images and not session.videos:
        st.error("No valid image or video files were found. Please upload supported formats.")
        return None

    gps_bytes = None
    gps_filename = "gps.csv"
    if gps_file:
        gps_bytes = gps_file.read()
        gps_filename = gps_file.name

    pipeline = InspectionPipeline(road_params)
    st.session_state["session_id"] = pipeline.session_id

    progress_bar = st.progress(0.0, text="Starting inspection…")

    def cb(frac: float, msg: str):
        progress_bar.progress(min(frac, 1.0), text=msg)

    with st.spinner("Running inspection pipeline…"):
        result = pipeline.run(
            session=session,
            gps_csv_bytes=gps_bytes,
            gps_csv_filename=gps_filename,
            video_start_datetime=video_start_dt,
            sample_fps=sample_fps,
            progress_callback=cb,
        )

    progress_bar.progress(1.0, text="✅ Inspection complete")
    return result


# ===========================================================================
# Results display
# ===========================================================================

def render_metrics(result: InspectionResult):
    st.markdown("## 📊 Inspection Results")

    counts = {t: 0 for t in DefectType}
    for d in result.defects:
        counts[d.defect_type] += 1

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    def metric_card(col, value, label, color="#1a3c5e"):
        col.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    metric_card(col1, len(result.source_files), "Files Analysed")
    metric_card(col2, result.total_frames_analysed, "Frames Analysed")
    metric_card(col3, result.defect_count, "Total Defects", "#c0392b" if result.defect_count > 0 else "#27ae60")
    metric_card(col4, counts[DefectType.MISSING], "Missing", "#c0392b")
    metric_card(col5, counts[DefectType.FADED], "Faded", "#e67e22")
    metric_card(col6, counts[DefectType.BROKEN], "Broken", "#d35400")

    # GPS status
    gps = result.gps_summary
    if gps["available"]:
        st.markdown(
            f'<span class="gps-badge-available">📍 GPS Available — {gps["mode"].upper()} '
            f'({gps["point_count"]} points, {gps["total_distance_m"]:.0f} m, {gps["avg_accuracy"]})</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="gps-badge-unavailable">📍 GPS Not Available</span>', unsafe_allow_html=True)

    # Errors / warnings
    if result.errors:
        with st.expander(f"⚠️ {len(result.errors)} Errors"):
            for e in result.errors:
                st.error(e)
    if result.warnings:
        with st.expander(f"ℹ️ {len(result.warnings)} Warnings"):
            for w in result.warnings:
                st.warning(w)


def render_defect_table(result: InspectionResult):
    if not result.defects:
        st.info("No potential defects were detected.")
        return

    st.markdown("### 🔍 Defect Table")

    # Filter
    filter_types = st.multiselect(
        "Filter by defect type",
        options=[t.value for t in DefectType],
        default=[t.value for t in DefectType],
    )

    filtered = [d for d in result.defects if d.defect_type.value in filter_types]

    rows = [d.to_dict() for d in filtered]
    df = pd.DataFrame(rows)

    if df.empty:
        st.info("No defects match the selected filter.")
        return

    # Display columns
    show_cols = ["ID", "Type", "Confidence", "Start Time (s)", "End Time (s)",
                 "Latitude", "Longitude", "GPS Accuracy (m)", "Review Status"]
    display_df = df[[c for c in show_cols if c in df.columns]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_defect_cards(result: InspectionResult):
    if not result.defects:
        return

    st.markdown("### 🗂️ Defect Detail & Manual Review")
    st.info(
        "ℹ️ Every AI result requires human review before any action is taken. "
        "Use the buttons below to confirm, reject, or flag for further review."
    )

    for d in result.defects:
        with st.expander(f"[{d.defect_id}] {d.defect_type.value}  —  conf={d.confidence:.2f}  |  {d.review_status}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Defect Type:** `{d.defect_type.value}`")
                st.markdown(f"**Confidence:** `{d.confidence:.2f}`")
                st.markdown(f"**Time:** {d.start_time_s:.1f}s – {d.end_time_s:.1f}s (duration: {d.duration_s:.1f}s)")
                if d.latitude:
                    st.markdown(f"**GPS:** {d.latitude:.6f}, {d.longitude:.6f}  (±{d.gps_accuracy_m or '?'} m)")
                if d.estimated_road_position:
                    st.markdown(f"**Road Position:** {d.estimated_road_position}")
                st.markdown(f"**Source File:** {d.source_file or '—'}")
                if d.applicable_standard:
                    st.markdown(f"**Applicable Standard:** {d.applicable_standard}")
                st.markdown(f"**Recommendation:** {d.recommendation}")
                if d.engineering_note:
                    st.markdown(f"**Engineering Note:** {d.engineering_note}")

            with col2:
                # Evidence image
                ev_img = _find_evidence_image(d.defect_id, result.session_id)
                if ev_img:
                    st.image(ev_img, caption="Annotated Evidence", use_column_width=True)
                else:
                    st.caption("No evidence image available")

            # Manual review buttons
            st.markdown("**Review Decision:**")
            r_col1, r_col2, r_col3 = st.columns(3)
            current = st.session_state.review_statuses.get(d.defect_id, d.review_status)

            if r_col1.button("✅ Confirm Defect", key=f"confirm_{d.defect_id}"):
                st.session_state.review_statuses[d.defect_id] = "Confirmed by inspector"
                d.review_status = "Confirmed by inspector"
                st.success("Marked as Confirmed")

            if r_col2.button("❌ Reject", key=f"reject_{d.defect_id}"):
                st.session_state.review_statuses[d.defect_id] = "Rejected by inspector"
                d.review_status = "Rejected by inspector"
                st.info("Marked as Rejected")

            if r_col3.button("🔎 Needs Review", key=f"review_{d.defect_id}"):
                st.session_state.review_statuses[d.defect_id] = "Needs further review"
                d.review_status = "Needs further review"
                st.warning("Flagged for further review")

            st.caption(f"Current status: {current}")


def _find_evidence_image(defect_id: str, session_id: str) -> Optional[Any]:
    from config.settings import PROCESSED_DIR
    ev_dir = PROCESSED_DIR / session_id / "evidence"
    if not ev_dir.exists():
        return None
    for f in ev_dir.glob(f"*{defect_id}_annotated.jpg"):
        return str(f)
    return None


def render_map(result: InspectionResult):
    st.markdown("### 🗺️ Inspection Map")
    if not result.gps_track or not result.gps_track.available:
        st.info("GPS unavailable — map not generated.")
        return

    if result.map_object is None:
        st.info("No GPS-tagged defects to display on map.")
        return

    try:
        from streamlit_folium import folium_static
        folium_static(result.map_object, width=900, height=500)
    except ImportError:
        try:
            from src.map_generator import map_to_html
            html = map_to_html(result.map_object)
            st.components.v1.html(html, height=500)
        except Exception as exc:
            st.warning(f"Map display error: {exc}")


def render_pdf_section(result: InspectionResult, road_params: RoadParameters):
    st.markdown("### 📄 Generate PDF Report")

    # Apply any manual review decisions
    for d in result.defects:
        if d.defect_id in st.session_state.review_statuses:
            d.review_status = st.session_state.review_statuses[d.defect_id]

    if st.button("🖨️ Generate PDF Report", type="primary", key="gen_pdf"):
        with st.spinner("Building PDF report…"):
            try:
                gen = ReportGenerator()
                pdf_bytes = gen.generate_bytes(
                    road_params=road_params,
                    defects=result.defects,
                    standards_result=result.standards_result,
                    gps_summary=result.gps_summary,
                    session_id=result.session_id,
                    source_files=result.source_files,
                    total_frames=result.total_frames_analysed,
                )
                filename = f"RoadMarkingAI_{road_params.road_name or 'report'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    key="download_pdf",
                )
                st.success("✅ PDF generated successfully!")
            except Exception as exc:
                st.error(f"PDF generation error: {exc}")
                logger.exception("PDF generation failed")


# ===========================================================================
# Standards display
# ===========================================================================

def render_standards(result: InspectionResult):
    if not result.standards_result:
        return
    sr = result.standards_result

    with st.expander("📚 Applicable IRC/MoRTH Standards (click to expand)"):
        if sr.applicable_references:
            for ref in sr.applicable_references:
                st.markdown(f"**{ref.get('id', '')} — {ref.get('standard', '')}**")
                st.caption(f"Section: {ref.get('section', '—')}")
                st.write(ref.get("applicability_notes", ""))
                st.info(f"📌 {ref.get('engineering_note', '')}")
                st.caption(f"Source: {ref.get('source_document', '—')}")
                st.divider()
        st.warning(
            "⚠️ " + sr.disclaimer + "\n\nAll standards data is for reference only. "
            "verification_required = TRUE for all entries."
        )


# ===========================================================================
# Disclaimer
# ===========================================================================

def render_disclaimer():
    st.markdown(
        f'<div class="disclaimer-box">⚠️ <b>IMPORTANT DISCLAIMER</b><br><br>{DISCLAIMER_TEXT.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# Main app
# ===========================================================================

def main():
    render_header()

    # Sidebar
    ui_params = render_sidebar()
    road_params = RoadParameters.from_ui_inputs(ui_params)
    sample_fps = ui_params["sample_fps"]

    # Input section
    media_files, gps_file, video_start_dt = render_input_section()

    # File preview
    if media_files:
        render_file_preview(media_files)
        for f in media_files:
            f.seek(0)  # reset after preview

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Process button
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("🚀 Run Inspection", type="primary", use_container_width=True)
    with col_info:
        if media_files:
            imgs = sum(1 for f in media_files if classify_extension(f.name) == "image")
            vids = sum(1 for f in media_files if classify_extension(f.name) == "video")
            st.info(f"Ready: {imgs} photo(s), {vids} video(s) | Sampling: {sample_fps} FPS")

    if run_btn:
        for f in media_files:
            f.seek(0)
        result = run_inspection(media_files, gps_file, video_start_dt, road_params, sample_fps)
        if result:
            st.session_state["inspection_result"] = result
            st.session_state["processing_done"] = True

    # Show results if available
    result: Optional[InspectionResult] = st.session_state.get("inspection_result")

    if result and st.session_state.get("processing_done"):
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        render_metrics(result)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📋 Defect Table", "🗂️ Detail & Review", "🗺️ Map", "📚 Standards", "📄 Report"]
        )

        with tab1:
            render_defect_table(result)
        with tab2:
            render_defect_cards(result)
        with tab3:
            render_map(result)
        with tab4:
            render_standards(result)
        with tab5:
            render_pdf_section(result, road_params)

    # Disclaimer always at bottom
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    render_disclaimer()


if __name__ == "__main__":
    main()
