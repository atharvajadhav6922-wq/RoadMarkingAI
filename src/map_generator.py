"""
src/map_generator.py

Generates a Folium interactive map of inspection results.

If GPS is unavailable, returns None and the UI shows a message.

Defect colour coding:
  green   — normal / no defect
  yellow  — faded
  red     — missing
  orange  — broken
  purple  — incorrect arrangement
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.defect_detector import DefectRecord, DefectType

logger = logging.getLogger(__name__)

# Colour mapping
DEFECT_COLOURS: Dict[DefectType, str] = {
    DefectType.MISSING: "red",
    DefectType.FADED: "beige",       # Folium's closest to yellow
    DefectType.BROKEN: "orange",
    DefectType.INCORRECT_ARRANGEMENT: "purple",
}


def generate_map(
    defects: List[DefectRecord],
    normal_points: Optional[List[Dict[str, float]]] = None,
) -> Optional[Any]:
    """
    Build and return a folium.Map object.

    Parameters:
        defects:       list of DefectRecord (must have latitude/longitude set)
        normal_points: list of {"lat": ..., "lon": ...} for non-defect GPS points

    Returns folium.Map or None if no GPS data available.
    """
    try:
        import folium
    except ImportError:
        logger.error("folium not installed — map cannot be generated")
        return None

    # Collect all points with GPS
    gps_defects = [d for d in defects if d.latitude is not None and d.longitude is not None]

    if not gps_defects and not normal_points:
        return None

    # Compute map centre
    all_lats = [d.latitude for d in gps_defects]
    all_lons = [d.longitude for d in gps_defects]
    if normal_points:
        all_lats += [p["lat"] for p in normal_points]
        all_lons += [p["lon"] for p in normal_points]

    centre_lat = sum(all_lats) / len(all_lats)
    centre_lon = sum(all_lons) / len(all_lons)

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=17,
        tiles="OpenStreetMap",
    )

    # Add normal GPS track (green dots)
    if normal_points:
        for pt in normal_points:
            folium.CircleMarker(
                location=[pt["lat"], pt["lon"]],
                radius=4,
                color="green",
                fill=True,
                fill_opacity=0.5,
                popup="Normal — no defect detected",
            ).add_to(m)

    # Add defect markers
    for defect in gps_defects:
        colour = DEFECT_COLOURS.get(defect.defect_type, "red")
        popup_html = (
            f"<b>Defect ID:</b> {defect.defect_id}<br>"
            f"<b>Type:</b> {defect.defect_type.value}<br>"
            f"<b>Confidence:</b> {defect.confidence:.2f}<br>"
            f"<b>Lat:</b> {defect.latitude:.6f}<br>"
            f"<b>Lon:</b> {defect.longitude:.6f}<br>"
            f"<b>GPS Accuracy:</b> {defect.gps_accuracy_m or 'unknown'} m<br>"
            f"<b>Time:</b> {defect.start_time_s:.1f}s – {defect.end_time_s:.1f}s<br>"
            f"<b>Review:</b> {defect.review_status}<br>"
            f"<b>Recommendation:</b> {defect.recommendation[:80]}..."
        )
        folium.Marker(
            location=[defect.latitude, defect.longitude],
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"{defect.defect_type.value} [{defect.defect_id}]",
            icon=folium.Icon(color=colour, icon="warning-sign", prefix="glyphicon"),
        ).add_to(m)

    return m


def map_to_html(m) -> str:
    """Render a folium map to an HTML string."""
    try:
        return m._repr_html_()
    except Exception:
        return "<p>Map rendering error.</p>"
