from __future__ import annotations

import enum
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dtflowcv.metrics import box_iou_matrix_np


# ── Track state ──────────────────────────────────────────────

class TrackStatus(enum.Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


class MotionState(enum.Enum):
    UNKNOWN = "unknown"
    MOVING = "moving"
    STATIONARY = "stationary"


# ── Kalman Filter (constant velocity model) ──────────────────

class KalmanBoxTracker:
    """Linear Kalman filter for bounding box tracking.

    State: [x_center, y_center, area, aspect_ratio, vx, vy, va, var]
    Measurement: [x_center, y_center, area, aspect_ratio]
    """

    _count = 0

    def __init__(self, bbox_xyxy: np.ndarray) -> None:
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count

        # Convert xyxy to [cx, cy, area, aspect_ratio]
        z = self._xyxy_to_z(bbox_xyxy)

        # State: 8-dimensional [cx, cy, s, r, vcx, vcy, vs, vr]
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = z

        # State transition matrix (constant velocity)
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = 1.0  # cx += vcx
        self.F[1, 5] = 1.0  # cy += vcy
        self.F[2, 6] = 1.0  # s  += vs
        self.F[3, 7] = 1.0  # r  += vr

        # Measurement matrix
        self.H = np.eye(4, 8, dtype=np.float64)

        # Covariance
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 1000.0  # High initial velocity uncertainty

        # Process noise
        self.Q = np.eye(8, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(4, dtype=np.float64)
        self.R[2, 2] *= 10.0  # Area measurement has higher noise
        self.R[3, 3] *= 10.0

        # Track statistics
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self) -> np.ndarray:
        """Advance state (predict next position)."""
        # Prevent negative area
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = 0 if self.time_since_update > 0 else self.hit_streak
        return self._x_to_xyxy()

    def update(self, bbox_xyxy: np.ndarray) -> None:
        """Update state with new measurement."""
        z = self._xyxy_to_z(bbox_xyxy)
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        I_KH = np.eye(8) - K @ self.H
        self.P = I_KH @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self) -> np.ndarray:
        """Current bounding box as [x1, y1, x2, y2]."""
        return self._x_to_xyxy()

    @staticmethod
    def _xyxy_to_z(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = w * h  # area
        r = w / max(h, 1e-6)  # aspect ratio
        return np.array([cx, cy, s, r], dtype=np.float64)

    def _x_to_xyxy(self) -> np.ndarray:
        cx, cy, s, r = self.x[:4]
        s = max(s, 1e-6)
        r = max(r, 1e-6)
        w = np.sqrt(s * r)
        h = s / max(w, 1e-6)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


# ── Track object ─────────────────────────────────────────────

@dataclass
class Track:
    """A tracked object with full lifecycle."""
    track_id: int
    class_id: int
    status: TrackStatus = TrackStatus.TENTATIVE
    motion: MotionState = MotionState.UNKNOWN
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    bbox_history: list[np.ndarray] = field(default_factory=list)
    _kalman: KalmanBoxTracker | None = None

    @property
    def last_bbox(self) -> np.ndarray | None:
        return self.bbox_history[-1] if self.bbox_history else None

    @property
    def centroid(self) -> tuple[float, float] | None:
        return self.trajectory[-1] if self.trajectory else None

    @property
    def age(self) -> int:
        return self._kalman.age if self._kalman else 0

    @property
    def avg_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    def is_moving(self, threshold: float = 5.0, window: int = 5) -> bool:
        """Check if object has moved significantly over last N frames."""
        if len(self.trajectory) < window:
            return self.motion == MotionState.MOVING
        recent = self.trajectory[-window:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        displacement = np.sqrt(dx * dx + dy * dy)
        return displacement > threshold


# ── SORT Tracker ─────────────────────────────────────────────

class SORTTracker:
    """Simple Online and Realtime Tracking (SORT).

    Manages object tracks using Kalman filter prediction + Hungarian
    assignment based on IoU cost matrix.

    Args:
        max_age: Frames to keep lost track alive before deletion.
        min_hits: Hits needed before track is confirmed.
        iou_threshold: Minimum IoU for detection-track matching.
        motion_threshold: Pixel displacement threshold for motion detection.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        motion_threshold: float = 5.0,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.motion_threshold = motion_threshold
        self._tracks: list[Track] = []
        self._frame_count = 0

    @property
    def tracks(self) -> list[Track]:
        """All active (non-deleted) tracks."""
        return [t for t in self._tracks if t.status != TrackStatus.DELETED]

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.status == TrackStatus.CONFIRMED]

    def update(
        self,
        detections: np.ndarray,
        scores: np.ndarray | None = None,
        class_ids: np.ndarray | None = None,
    ) -> list[Track]:
        """Process new frame detections.

        Args:
            detections: (N, 4) array of [x1, y1, x2, y2] bounding boxes.
            scores: (N,) confidence scores.
            class_ids: (N,) class IDs.

        Returns:
            List of active tracks after update.
        """
        self._frame_count += 1

        if scores is None:
            scores = np.ones(len(detections), dtype=np.float32)
        if class_ids is None:
            class_ids = np.zeros(len(detections), dtype=np.int32)

        # 1. Predict all existing tracks
        predicted_boxes = []
        active_tracks = [t for t in self._tracks if t.status != TrackStatus.DELETED]
        for track in active_tracks:
            if track._kalman is not None:
                pred = track._kalman.predict()
                predicted_boxes.append(pred)
            else:
                predicted_boxes.append(np.zeros(4))

        # 2. Match detections to tracks using IoU
        matched, unmatched_dets, unmatched_trks = self._match(
            detections, np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4)),
        )

        # 3. Update matched tracks
        for det_idx, trk_idx in matched:
            track = active_tracks[trk_idx]
            bbox = detections[det_idx]
            if track._kalman is not None:
                track._kalman.update(bbox)
            track.bbox_history.append(bbox.copy())
            track.scores.append(float(scores[det_idx]))
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            track.trajectory.append((float(cx), float(cy)))

            if track._kalman is not None and track._kalman.hits >= self.min_hits:
                track.status = TrackStatus.CONFIRMED

            # Update motion state
            track.motion = (
                MotionState.MOVING if track.is_moving(self.motion_threshold)
                else MotionState.STATIONARY
            )

        # 4. Mark unmatched tracks as lost
        for trk_idx in unmatched_trks:
            track = active_tracks[trk_idx]
            if track._kalman is not None and track._kalman.time_since_update > self.max_age:
                track.status = TrackStatus.DELETED
            elif track.status == TrackStatus.CONFIRMED:
                track.status = TrackStatus.LOST

        # 5. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox = detections[det_idx]
            kalman = KalmanBoxTracker(bbox)
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            track = Track(
                track_id=kalman.id,
                class_id=int(class_ids[det_idx]),
                _kalman=kalman,
                trajectory=[(float(cx), float(cy))],
                scores=[float(scores[det_idx])],
                bbox_history=[bbox.copy()],
            )
            self._tracks.append(track)

        # 6. Cleanup deleted tracks
        self._tracks = [t for t in self._tracks if t.status != TrackStatus.DELETED]

        return self.tracks

    def _match(
        self,
        detections: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian matching via IoU cost matrix."""
        n_det = len(detections)
        n_trk = len(predictions)

        if n_det == 0:
            return [], [], list(range(n_trk))
        if n_trk == 0:
            return [], list(range(n_det)), []

        # IoU matrix: (n_det, n_trk)
        iou_matrix = box_iou_matrix_np(
            detections.astype(np.float32),
            predictions.astype(np.float32),
        )

        # Greedy assignment (simpler than full Hungarian, adequate for SORT)
        matched = []
        used_dets: set[int] = set()
        used_trks: set[int] = set()

        # Sort by IoU descending to match best pairs first
        if iou_matrix.size > 0:
            flat_indices = np.argsort(-iou_matrix.ravel())
            for flat_idx in flat_indices:
                det_idx = int(flat_idx // n_trk)
                trk_idx = int(flat_idx % n_trk)
                if det_idx in used_dets or trk_idx in used_trks:
                    continue
                if iou_matrix[det_idx, trk_idx] < self.iou_threshold:
                    break  # All remaining IoUs are lower
                matched.append((det_idx, trk_idx))
                used_dets.add(det_idx)
                used_trks.add(trk_idx)

        unmatched_dets = [i for i in range(n_det) if i not in used_dets]
        unmatched_trks = [i for i in range(n_trk) if i not in used_trks]

        return matched, unmatched_dets, unmatched_trks

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._frame_count = 0
        KalmanBoxTracker._count = 0
