"""
src/road_roi.py

Road Region-of-Interest (ROI) extraction.

The camera is assumed to be mounted facing forward from a moving vehicle.
A trapezoidal ROI excludes sky, buildings, roadside objects, and the vehicle
bonnet, focusing detection on the road surface ahead.

All fractions are configurable in config/settings.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    ROI_BOTTOM_LEFT_X_FRAC,
    ROI_BOTTOM_RIGHT_X_FRAC,
    ROI_BOTTOM_Y_FRAC,
    ROI_TOP_LEFT_X_FRAC,
    ROI_TOP_RIGHT_X_FRAC,
    ROI_TOP_Y_FRAC,
)


@dataclass
class ROIConfig:
    """Trapezoidal ROI configuration expressed as frame fractions."""
    top_y: float = ROI_TOP_Y_FRAC
    bottom_y: float = ROI_BOTTOM_Y_FRAC
    top_left_x: float = ROI_TOP_LEFT_X_FRAC
    top_right_x: float = ROI_TOP_RIGHT_X_FRAC
    bottom_left_x: float = ROI_BOTTOM_LEFT_X_FRAC
    bottom_right_x: float = ROI_BOTTOM_RIGHT_X_FRAC


class RoadROI:
    """
    Applies a perspective-aware trapezoidal mask to a frame.

    The trapezoid is defined by four corners:
      - Top-left, Top-right (near the horizon)
      - Bottom-left, Bottom-right (near the camera/bonnet)

    Everything outside the trapezoid is masked to black so that
    downstream detectors only see road pixels.
    """

    def __init__(self, config: Optional[ROIConfig] = None):
        self.config = config or ROIConfig()

    def compute_vertices(self, height: int, width: int) -> np.ndarray:
        """Return trapezoid vertices as (4, 2) int32 array."""
        cfg = self.config
        tl = (int(cfg.top_left_x * width),    int(cfg.top_y * height))
        tr = (int(cfg.top_right_x * width),   int(cfg.top_y * height))
        br = (int(cfg.bottom_right_x * width), int(cfg.bottom_y * height))
        bl = (int(cfg.bottom_left_x * width),  int(cfg.bottom_y * height))
        return np.array([tl, tr, br, bl], dtype=np.int32)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return frame with everything outside the ROI set to black."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        vertices = self.compute_vertices(h, w)
        cv2.fillPoly(mask, [vertices], 255)
        # Apply mask to all channels
        if len(frame.shape) == 3:
            mask_3ch = cv2.merge([mask, mask, mask])
            return cv2.bitwise_and(frame, mask_3ch)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def apply_with_debug(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (masked_frame, debug_frame).
        Debug frame shows the ROI polygon drawn on the original image.
        """
        masked = self.apply(frame)
        debug = frame.copy()
        h, w = frame.shape[:2]
        vertices = self.compute_vertices(h, w)
        cv2.polylines(debug, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)
        return masked, debug

    def get_perspective_transform(
        self, height: int, width: int, output_size: Tuple[int, int] = (640, 480)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute perspective warp matrices (birds-eye-view transform).
        Returns (M_warp, M_unwarp).
        Useful for calibrated distance estimation.
        """
        cfg = self.config
        src = np.float32([
            [cfg.top_left_x * width,    cfg.top_y * height],
            [cfg.top_right_x * width,   cfg.top_y * height],
            [cfg.bottom_right_x * width, cfg.bottom_y * height],
            [cfg.bottom_left_x * width,  cfg.bottom_y * height],
        ])
        ow, oh = output_size
        dst = np.float32([
            [0, 0],
            [ow, 0],
            [ow, oh],
            [0, oh],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        return M, M_inv

    def warp_to_bev(
        self,
        frame: np.ndarray,
        output_size: Tuple[int, int] = (640, 480),
    ) -> np.ndarray:
        """Return a bird's-eye-view warped version of the ROI region."""
        h, w = frame.shape[:2]
        M, _ = self.get_perspective_transform(h, w, output_size)
        return cv2.warpPerspective(frame, M, output_size)
