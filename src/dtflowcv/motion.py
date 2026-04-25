from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class BackgroundSubtractor:
    """Motion detection via background subtraction (MOG2).

    Uses OpenCV's MOG2 algorithm to build a background model and
    detect foreground (moving) pixels.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        learning_rate: float = -1.0,
        min_area: int = 500,
        kernel_size: int = 5,
    ) -> None:
        self._history = history
        self._var_threshold = var_threshold
        self._detect_shadows = detect_shadows
        self._learning_rate = learning_rate
        self._min_area = min_area
        self._kernel_size = kernel_size
        self._subtractor: Any = None

    def _ensure_subtractor(self) -> None:
        if self._subtractor is None:
            import cv2
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self._history,
                varThreshold=self._var_threshold,
                detectShadows=self._detect_shadows,
            )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply background subtraction. Returns binary motion mask."""
        import cv2

        self._ensure_subtractor()
        fg_mask = self._subtractor.apply(frame, learningRate=self._learning_rate)

        # Remove shadows (value 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._kernel_size, self._kernel_size))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        return fg_mask

    def detect_motion_regions(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect motion regions as bounding boxes [x1, y1, x2, y2]."""
        import cv2

        mask = self.apply(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, x + w, y + h))

        return regions

    def reset(self) -> None:
        self._subtractor = None


class FrameDifferencer:
    """Motion detection via frame differencing.

    Compares consecutive frames to detect motion.
    """

    def __init__(
        self,
        threshold: int = 25,
        min_area: int = 500,
        blur_size: int = 21,
    ) -> None:
        self._threshold = threshold
        self._min_area = min_area
        self._blur_size = blur_size
        self._prev_gray: np.ndarray | None = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Compute motion mask from frame difference. Returns binary mask."""
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._blur_size, self._blur_size), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return np.zeros(gray.shape, dtype=np.uint8)

        diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(diff, self._threshold, 255, cv2.THRESH_BINARY)

        # Dilate to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        self._prev_gray = gray
        return thresh

    def detect_motion_regions(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect motion regions as [x1, y1, x2, y2]."""
        import cv2

        mask = self.apply(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            if cv2.contourArea(contour) < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, x + w, y + h))
        return regions

    def reset(self) -> None:
        self._prev_gray = None


# ── Line Crossing Counter ────────────────────────────────────

@dataclass
class LineCrossing:
    """Virtual line crossing detector.

    Counts objects (by track centroid) crossing a line segment
    in each direction (A→B and B→A).
    """
    line_start: tuple[int, int]
    line_end: tuple[int, int]
    count_ab: int = 0   # Crossing from side A to side B
    count_ba: int = 0   # Crossing from side B to side A
    _prev_sides: dict[int, int] = field(default_factory=dict)

    def _side(self, point: tuple[float, float]) -> int:
        """Which side of the line is the point on? Returns +1 or -1."""
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        return 1 if cross >= 0 else -1

    def update(self, track_id: int, centroid: tuple[float, float]) -> str | None:
        """Update with new centroid. Returns 'ab', 'ba', or None."""
        side = self._side(centroid)
        prev = self._prev_sides.get(track_id)
        self._prev_sides[track_id] = side

        if prev is not None and prev != side:
            if prev == -1 and side == 1:
                self.count_ab += 1
                return "ab"
            else:
                self.count_ba += 1
                return "ba"
        return None

    def total(self) -> int:
        return self.count_ab + self.count_ba

    def reset(self) -> None:
        self.count_ab = 0
        self.count_ba = 0
        self._prev_sides.clear()


# ── Zone Monitor ─────────────────────────────────────────────

@dataclass
class ZoneMonitor:
    """ROI polygon zone monitor.

    Counts objects currently inside a polygonal region.
    """
    polygon: list[tuple[int, int]]  # Vertices of the polygon
    _inside: set[int] = field(default_factory=set)
    enter_count: int = 0
    exit_count: int = 0

    def _point_in_polygon(self, point: tuple[float, float]) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        x, y = point
        n = len(self.polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def update(self, track_id: int, centroid: tuple[float, float]) -> str | None:
        """Update with track centroid.

        Returns:
            'enter' if object just entered the zone,
            'exit' if object just left,
            None otherwise.
        """
        is_inside = self._point_in_polygon(centroid)
        was_inside = track_id in self._inside

        if is_inside and not was_inside:
            self._inside.add(track_id)
            self.enter_count += 1
            return "enter"
        elif not is_inside and was_inside:
            self._inside.discard(track_id)
            self.exit_count += 1
            return "exit"
        return None

    @property
    def current_count(self) -> int:
        return len(self._inside)

    def reset(self) -> None:
        self._inside.clear()
        self.enter_count = 0
        self.exit_count = 0


class MotionDetector:
    """Combined motion detector using both background subtraction and frame differencing.

    Provides a unified API for motion detection, combining the outputs
    of both methods for robustness.
    """

    def __init__(
        self,
        use_mog2: bool = True,
        use_frame_diff: bool = True,
        mog2_threshold: float = 16.0,
        diff_threshold: int = 25,
        min_area: int = 500,
    ) -> None:
        self._bg_sub = BackgroundSubtractor(
            var_threshold=mog2_threshold, min_area=min_area,
        ) if use_mog2 else None

        self._frame_diff = FrameDifferencer(
            threshold=diff_threshold, min_area=min_area,
        ) if use_frame_diff else None

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """Detect motion in frame.

        Returns:
            Tuple of (combined_mask, motion_regions) where motion_regions
            are [x1, y1, x2, y2] bounding boxes.
        """
        import cv2

        masks = []
        if self._bg_sub is not None:
            masks.append(self._bg_sub.apply(frame))
        if self._frame_diff is not None:
            masks.append(self._frame_diff.apply(frame))

        if not masks:
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8), []

        # Combine masks with OR
        combined = masks[0]
        for mask in masks[1:]:
            combined = cv2.bitwise_or(combined, mask)

        # Find contours for regions
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            if cv2.contourArea(contour) < (self._bg_sub._min_area if self._bg_sub else 500):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, x + w, y + h))

        return combined, regions

    def reset(self) -> None:
        if self._bg_sub:
            self._bg_sub.reset()
        if self._frame_diff:
            self._frame_diff.reset()
