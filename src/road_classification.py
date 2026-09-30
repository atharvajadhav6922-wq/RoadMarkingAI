"""
src/road_classification.py

Maps user-supplied road classification to internal identifiers used by the
standards engine and defect detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ------------------------------------------------------------------
# Road classification constants (match irc35_rules.json road_type values)
# ------------------------------------------------------------------
ROAD_CLASSES = {
    "Urban Arterial": "urban_arterial",
    "Urban Sub-Arterial": "urban_sub_arterial",
    "Collector / Local Road": "collector_local",
    "Urban Municipal Road": "urban_municipal",
    "Other": "other",
}

PAVEMENT_TYPES = {
    "Asphalt / Bituminous": "asphalt",
    "Concrete": "concrete",
    "Other": "other",
}

EXPECTED_MARKINGS = {
    "Auto / Hybrid": "auto",
    "Centre Line": "centre_line",
    "Lane Lines": "lane_line",
    "Centre + Lane Lines": "centre_and_lane",
    "Edge Lines": "edge_line",
    "User Defined": "user_defined",
    "No Marking Specified": "none",
}


@dataclass
class RoadParameters:
    """All user-supplied road geometry and classification parameters."""
    road_name: str = ""
    authority: str = ""
    inspection_date: str = ""
    inspector_name: str = ""
    road_class_display: str = "Urban Arterial"
    road_class: str = "urban_arterial"
    divided: bool = False
    num_lanes: int = 2
    carriageway_width_m: Optional[float] = None
    lane_width_m: Optional[float] = None
    shoulder_width_m: Optional[float] = None
    pavement_type: str = "asphalt"
    inspection_direction: str = ""
    expected_marking_display: str = "Auto / Hybrid"
    expected_marking: str = "auto"

    @property
    def geometry_label(self) -> str:
        div = "Divided" if self.divided else "Undivided"
        return f"{div}, {self.num_lanes}-lane, {self.road_class_display}"

    @classmethod
    def from_ui_inputs(cls, ui: dict) -> "RoadParameters":
        """Build from Streamlit sidebar widget values."""
        rc_display = ui.get("road_class_display", "Urban Arterial")
        em_display = ui.get("expected_marking_display", "Auto / Hybrid")
        pt_display = ui.get("pavement_type_display", "Asphalt / Bituminous")
        return cls(
            road_name=ui.get("road_name", ""),
            authority=ui.get("authority", ""),
            inspection_date=ui.get("inspection_date", ""),
            inspector_name=ui.get("inspector_name", ""),
            road_class_display=rc_display,
            road_class=ROAD_CLASSES.get(rc_display, "urban_arterial"),
            divided=ui.get("divided", False),
            num_lanes=ui.get("num_lanes", 2),
            carriageway_width_m=ui.get("carriageway_width_m"),
            lane_width_m=ui.get("lane_width_m"),
            shoulder_width_m=ui.get("shoulder_width_m"),
            pavement_type=PAVEMENT_TYPES.get(pt_display, "asphalt"),
            inspection_direction=ui.get("inspection_direction", ""),
            expected_marking_display=em_display,
            expected_marking=EXPECTED_MARKINGS.get(em_display, "auto"),
        )
