"""
src/standards_engine.py

Queries the IRC/MoRTH standards database and returns applicable references
for the given road parameters.

IMPORTANT:
  - No dimensions are invented.
  - All entries carry verification_required=true.
  - The engine only advises which standards are applicable and directs
    the engineer to verify against the official document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import BASE_DIR

RULES_PATH = BASE_DIR / "standards" / "irc35_rules.json"
CONFIG_PATH = BASE_DIR / "config" / "standards_config.json"


@dataclass
class StandardsResult:
    applicable_references: List[Dict[str, Any]] = field(default_factory=list)
    engineering_note: str = ""
    verification_required: bool = True
    disclaimer: str = (
        "Applicable IRC/MoRTH provision requires verification against the current "
        "official edition/amendments and sanctioned road layout."
    )

    @property
    def reference_ids(self) -> List[str]:
        return [r.get("id", "") for r in self.applicable_references]

    @property
    def standards_list(self) -> str:
        return "; ".join(r.get("standard", "") for r in self.applicable_references)


class StandardsEngine:
    """
    Queries irc35_rules.json for rules applicable to the supplied road parameters.

    Usage:
        engine = StandardsEngine()
        result = engine.query(
            road_class="urban_arterial",
            marking_type="centre_line",
            divided=False,
        )
    """

    def __init__(self):
        self._rules: List[Dict[str, Any]] = []
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(RULES_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
                self._rules = data.get("rules", [])
        except Exception as exc:
            self._rules = []
            import logging
            logging.getLogger(__name__).error(f"StandardsEngine: cannot load rules: {exc}")

        try:
            with open(CONFIG_PATH, encoding="utf-8") as fh:
                self._config = json.load(fh)
        except Exception:
            self._config = {}

    def query(
        self,
        road_class: str,
        marking_type: str = "any",
        divided: bool = False,
    ) -> StandardsResult:
        """
        Return applicable IRC/MoRTH references for the given road + marking type.

        Parameters:
            road_class:    internal road class ID (e.g. "urban_arterial")
            marking_type:  internal marking type (e.g. "centre_line", "lane_line")
            divided:       True if the road is divided (dual carriageway)
        """
        div_str = "divided" if divided else "undivided"
        matches: List[Dict[str, Any]] = []

        for rule in self._rules:
            road_types = rule.get("road_type", [])
            r_marking = rule.get("marking_type", "any")
            r_div = rule.get("divided_undivided", "any")

            type_match = (
                "any" in road_types
                or road_class in road_types
                or road_class.replace("_", " ") in road_types
            )
            marking_match = (
                marking_type in ("any", "auto")
                or r_marking == marking_type
                or r_marking == "general_road_marking"
            )
            div_match = r_div in ("any", div_str)

            if type_match and marking_match and div_match:
                matches.append(rule)

        engineering_notes = list({r.get("engineering_note", "") for r in matches if r.get("engineering_note")})
        note = " | ".join(engineering_notes) if engineering_notes else (
            "No specific rule found. Verify against IRC:35-2015 and MoRTH Section 803 for this road type."
        )

        return StandardsResult(
            applicable_references=matches,
            engineering_note=note,
            verification_required=True,
            disclaimer=self._config.get("disclaimer", StandardsResult.disclaimer),
        )

    def get_all_references(self) -> List[Dict[str, Any]]:
        """Return all rules in the database."""
        return list(self._rules)
