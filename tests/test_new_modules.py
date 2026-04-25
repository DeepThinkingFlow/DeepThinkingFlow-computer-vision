"""Tests for tracking module — SORT tracker with Kalman filter."""
from __future__ import annotations

import numpy as np
import pytest

from dtflowcv.tracking import KalmanBoxTracker, MotionState, SORTTracker, Track, TrackStatus


class TestKalmanBoxTracker:
    def test_init_and_state(self):
        bbox = np.array([10.0, 20.0, 50.0, 60.0])
        kbt = KalmanBoxTracker(bbox)
        state = kbt.get_state()
        # Should be close to original box
        np.testing.assert_allclose(state, bbox, atol=1.0)

    def test_predict_advances_state(self):
        bbox = np.array([10.0, 20.0, 50.0, 60.0])
        kbt = KalmanBoxTracker(bbox)
        pred = kbt.predict()
        assert kbt.age == 1
        assert kbt.time_since_update == 1

    def test_update_resets_time(self):
        bbox = np.array([10.0, 20.0, 50.0, 60.0])
        kbt = KalmanBoxTracker(bbox)
        kbt.predict()
        assert kbt.time_since_update == 1
        kbt.update(np.array([12.0, 22.0, 52.0, 62.0]))
        assert kbt.time_since_update == 0
        assert kbt.hits == 2


class TestSORTTracker:
    def test_new_detections_create_tracks(self):
        tracker = SORTTracker(min_hits=1)
        dets = np.array([[10, 20, 50, 60], [100, 200, 150, 260]], dtype=np.float32)
        tracks = tracker.update(dets)
        assert len(tracks) == 2

    def test_consistent_track_ids(self):
        tracker = SORTTracker(min_hits=1, iou_threshold=0.1)
        dets1 = np.array([[10, 20, 50, 60]], dtype=np.float32)
        tracks1 = tracker.update(dets1)
        id1 = tracks1[0].track_id

        # Slightly moved
        dets2 = np.array([[12, 22, 52, 62]], dtype=np.float32)
        tracks2 = tracker.update(dets2)
        assert any(t.track_id == id1 for t in tracks2)

    def test_empty_detection_keeps_tracks(self):
        tracker = SORTTracker(min_hits=1, max_age=5)
        dets = np.array([[10, 20, 50, 60]], dtype=np.float32)
        tracker.update(dets)
        # No detections
        tracks = tracker.update(np.empty((0, 4), dtype=np.float32))
        assert len(tracks) >= 1  # Track still alive (max_age=5)

    def test_motion_state_detection(self):
        tracker = SORTTracker(min_hits=1, motion_threshold=3.0)
        # Object moving rightward
        for i in range(10):
            dets = np.array([[10 + i * 5, 20, 50 + i * 5, 60]], dtype=np.float32)
            tracker.update(dets)
        tracks = tracker.tracks
        assert len(tracks) >= 1
        # After 10 frames of movement, should be MOVING
        moving = [t for t in tracks if t.motion == MotionState.MOVING]
        assert len(moving) >= 1

    def test_trajectory_recorded(self):
        tracker = SORTTracker(min_hits=1)
        for i in range(5):
            dets = np.array([[10 + i * 2, 20, 50 + i * 2, 60]], dtype=np.float32)
            tracker.update(dets)
        tracks = tracker.tracks
        assert len(tracks) >= 1
        assert len(tracks[0].trajectory) >= 5

    def test_reset_clears_state(self):
        tracker = SORTTracker()
        tracker.update(np.array([[10, 20, 50, 60]], dtype=np.float32))
        assert len(tracker.tracks) > 0
        tracker.reset()
        assert len(tracker.tracks) == 0


class TestMotionDetection:
    def test_line_crossing_counter(self):
        from dtflowcv.motion import LineCrossing
        lc = LineCrossing(line_start=(0, 100), line_end=(200, 100))
        # Object moves from above to below the line
        lc.update(1, (50.0, 80.0))   # Above
        result = lc.update(1, (50.0, 120.0))  # Below
        assert result is not None
        assert lc.total() == 1

    def test_zone_monitor(self):
        from dtflowcv.motion import ZoneMonitor
        zone = ZoneMonitor(polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
        result = zone.update(1, (50.0, 50.0))  # Inside
        assert result == "enter"
        assert zone.current_count == 1

        result = zone.update(1, (150.0, 150.0))  # Outside
        assert result == "exit"
        assert zone.current_count == 0

    def test_zone_point_in_polygon(self):
        from dtflowcv.motion import ZoneMonitor
        zone = ZoneMonitor(polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
        assert zone._point_in_polygon((50, 50)) is True
        assert zone._point_in_polygon((150, 150)) is False
        assert zone._point_in_polygon((0, 0)) is False  # Edge case


class TestDataCard:
    def test_build_and_write_dataset_card(self, tmp_path):
        from dtflowcv.data_card import build_dataset_card, write_dataset_card

        # Create minimal dataset
        img_dir = tmp_path / "images" / "train"
        lbl_dir = tmp_path / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Fake image
        from PIL import Image
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        img.save(img_dir / "test.jpg")
        (lbl_dir / "test.txt").write_text("0 0.5 0.5 0.3 0.3\n")

        card = build_dataset_card(tmp_path, "test-dataset", ["person", "car"])
        assert card.image_count == 1
        assert card.label_count == 1
        assert card.annotation_count == 1
        assert card.sha256 != ""
        assert card.class_distribution["person"] == 1

        paths = write_dataset_card(card, tmp_path / "output")
        assert Path(paths["json"]).exists()
        assert Path(paths["markdown"]).exists()
        assert Path(paths["sha256"]).exists()

    def test_verify_integrity(self, tmp_path):
        from dtflowcv.data_card import compute_dataset_hash, verify_dataset_integrity

        (tmp_path / "test.txt").write_text("hello\n")
        h = compute_dataset_hash(tmp_path)
        result = verify_dataset_integrity(tmp_path, h)
        assert result["verified"] is True

        result2 = verify_dataset_integrity(tmp_path, "wrong_hash")
        assert result2["verified"] is False


class TestEnhancedMetrics:
    def test_confusion_matrix_shape(self):
        from dtflowcv.metrics import DetectionPrediction, DetectionTarget, confusion_matrix

        targets = [
            DetectionTarget("img1", 0, (10, 10, 50, 50)),
            DetectionTarget("img1", 1, (60, 60, 100, 100)),
        ]
        predictions = [
            DetectionPrediction("img1", 0, (12, 12, 48, 48), 0.9),
            DetectionPrediction("img1", 0, (62, 62, 98, 98), 0.8),  # Class confusion: GT=1, pred=0
        ]
        result = confusion_matrix(targets, predictions, 2, iou_threshold=0.3)
        matrix = result["matrix"]
        assert len(matrix) == 3  # 2 classes + background
        assert len(matrix[0]) == 3

    def test_per_class_detail_in_map(self):
        from dtflowcv.metrics import DetectionPrediction, DetectionTarget, map_at_iou

        targets = [
            DetectionTarget("img1", 0, (10, 10, 50, 50)),
        ]
        predictions = [
            DetectionPrediction("img1", 0, (10, 10, 50, 50), 0.9),
        ]
        result = map_at_iou(targets, predictions, 2, iou_threshold=0.5)
        assert "class_detail" in result
        detail = result["class_detail"]["0"]
        assert "precision" in detail
        assert "recall" in detail
        assert "f1" in detail
        assert detail["precision"] == 1.0
        assert detail["recall"] == 1.0

    def test_precision_recall_curve(self):
        from dtflowcv.metrics import DetectionPrediction, DetectionTarget, precision_recall_curve

        targets = [DetectionTarget("img1", 0, (10, 10, 50, 50))]
        predictions = [DetectionPrediction("img1", 0, (10, 10, 50, 50), 0.9)]
        curve = precision_recall_curve(targets, predictions, 0, n_points=11)
        assert len(curve["recall"]) == 11
        assert len(curve["precision"]) == 11
