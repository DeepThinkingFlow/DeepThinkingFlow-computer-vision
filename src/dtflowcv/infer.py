from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from dtflowcv.config import write_json


def infer_video(
    video_source: str | Path | int,
    problem_path: str | Path,
    model_path: str | Path = "yolov8n.pt",
    output_video: str | Path | None = None,
    output_json: str | Path | None = None,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | int | None = None,
    sample_fps: float | None = None,
    max_frames: int | None = None,
    enable_tracking: bool = True,
    tracker_max_age: int = 30,
    tracker_min_hits: int = 3,
    tracker_iou: float = 0.3,
    draw_trajectory: bool = True,
) -> dict[str, Any]:
    """Full video inference pipeline: detect → track → annotate → write.

    Args:
        video_source: Video file, RTSP URL, or webcam device ID.
        problem_path: Problem YAML for class names.
        model_path: YOLO checkpoint.
        output_video: Output annotated video path (None = no video output).
        output_json: Output per-frame JSON predictions (None = no JSON).
        conf: Detection confidence threshold.
        iou: NMS IoU threshold.
        device: Inference device.
        sample_fps: Downsample video FPS.
        max_frames: Max frames to process.
        enable_tracking: Enable SORT tracking.
        draw_trajectory: Draw trajectory trails.

    Returns:
        Summary dict with timing, counts, output paths.
    """
    from dtflowcv.config import load_yaml
    from dtflowcv.predict import model_class_map
    from dtflowcv.specs import class_names, validate_problem_spec
    from dtflowcv.video import VideoReader, VideoWriter

    problem = load_yaml(problem_path)
    errors = validate_problem_spec(problem)
    if errors:
        return {"status": "blocked", "errors": errors}

    names = class_names(problem)

    try:
        from ultralytics import YOLO
    except ImportError:
        return {"status": "blocked", "reason": "ultralytics not installed"}

    model = YOLO(str(model_path))
    cmap = model_class_map(getattr(model, "names", {}), names)

    tracker = None
    if enable_tracking:
        from dtflowcv.tracking import SORTTracker
        tracker = SORTTracker(
            max_age=tracker_max_age,
            min_hits=tracker_min_hits,
            iou_threshold=tracker_iou,
        )

    writer = None
    if output_video is not None:
        writer = VideoWriter(output_video)

    frame_results: list[dict[str, Any]] = []
    total_detections = 0
    total_tracks = 0
    t_start = time.perf_counter()

    with VideoReader(video_source, sample_fps=sample_fps, max_frames=max_frames) as reader:
        native_fps = reader.fps

        for frame in reader:
            # Detect
            result = model.predict(frame.image, conf=conf, iou=iou, device=device, verbose=False)[0]

            det_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            det_scores = result.boxes.conf.cpu().numpy().astype(np.float32)
            det_cls = result.boxes.cls.cpu().numpy().astype(np.int32)

            # Map to problem classes
            keep = []
            mapped_cls = []
            for i in range(len(det_cls)):
                target_cid = cmap.get(int(det_cls[i]))
                if target_cid is not None:
                    keep.append(i)
                    mapped_cls.append(target_cid)

            if keep:
                det_boxes = det_boxes[keep]
                det_scores = det_scores[keep]
                det_cls = np.array(mapped_cls, dtype=np.int32)
            else:
                det_boxes = np.empty((0, 4), dtype=np.float32)
                det_scores = np.empty(0, dtype=np.float32)
                det_cls = np.empty(0, dtype=np.int32)

            total_detections += len(det_boxes)

            # Track
            tracks = []
            if tracker is not None and len(det_boxes) > 0:
                tracks = tracker.update(det_boxes, det_scores, det_cls)
            elif tracker is not None:
                tracks = tracker.update(np.empty((0, 4), dtype=np.float32))
            total_tracks = max(total_tracks, len(tracks))

            # Build per-frame record
            frame_data: dict[str, Any] = {
                "frame_index": frame.meta.index,
                "timestamp_ms": frame.meta.timestamp_ms,
                "detections": len(det_boxes),
            }
            if tracker is not None:
                frame_data["active_tracks"] = len(tracks)
                frame_data["tracks"] = [
                    {
                        "id": t.track_id,
                        "class_id": t.class_id,
                        "class_name": names[t.class_id] if t.class_id < len(names) else f"cls_{t.class_id}",
                        "bbox": t.last_bbox.tolist() if t.last_bbox is not None else [],
                        "motion": t.motion.value,
                        "score": round(t.avg_score, 3),
                    }
                    for t in tracks
                ]
            frame_results.append(frame_data)

            # Draw
            if writer is not None:
                from dtflowcv.visualize import draw_detections, draw_tracking

                annotated = frame.image.copy()
                if tracker is not None and tracks:
                    annotated = draw_tracking(annotated, tracks, class_names=names,
                                              draw_trajectory=draw_trajectory)
                elif len(det_boxes) > 0:
                    annotated = draw_detections(annotated, det_boxes, det_cls, det_scores, class_names=names)

                # FPS overlay
                import cv2
                elapsed = time.perf_counter() - t_start
                cur_fps = (frame.meta.index + 1) / max(elapsed, 1e-6)
                cv2.putText(annotated, f"FPS: {cur_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                writer.write(annotated)

    elapsed = time.perf_counter() - t_start
    frames_processed = len(frame_results)

    if writer is not None:
        writer.close()

    if output_json is not None:
        out_json = Path(output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_json, {
            "source": str(video_source),
            "model": str(model_path),
            "frames": frame_results,
        })

    return {
        "status": "ok",
        "source": str(video_source),
        "model": str(model_path),
        "frames_processed": frames_processed,
        "total_detections": total_detections,
        "peak_active_tracks": total_tracks,
        "elapsed_seconds": round(elapsed, 2),
        "avg_fps": round(frames_processed / max(elapsed, 1e-6), 1),
        "native_fps": native_fps,
        "output_video": str(output_video) if output_video else None,
        "output_json": str(output_json) if output_json else None,
        "tracking_enabled": enable_tracking,
    }


def infer_images(
    images_dir: str | Path,
    problem_path: str | Path,
    model_path: str | Path = "yolov8n.pt",
    output_dir: str | Path = "artifacts/annotated",
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | int | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Batch inference on images with annotated output.

    Saves annotated images to output_dir.
    """
    from dtflowcv.config import load_yaml
    from dtflowcv.predict import model_class_map
    from dtflowcv.specs import class_names, validate_problem_spec
    from dtflowcv.yolo import iter_images

    problem = load_yaml(problem_path)
    errors = validate_problem_spec(problem)
    if errors:
        return {"status": "blocked", "errors": errors}

    names = class_names(problem)

    try:
        from ultralytics import YOLO
    except ImportError:
        return {"status": "blocked", "reason": "ultralytics not installed"}

    model = YOLO(str(model_path))
    cmap = model_class_map(getattr(model, "names", {}), names)

    images_root = Path(images_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(images_root)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    import cv2
    from dtflowcv.visualize import draw_detections

    total = 0
    total_dets = 0
    t_start = time.perf_counter()

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        result = model.predict(img, conf=conf, iou=iou, device=device, verbose=False)[0]
        det_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        det_scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        det_cls = result.boxes.cls.cpu().numpy().astype(np.int32)

        keep = []
        mapped = []
        for i in range(len(det_cls)):
            t = cmap.get(int(det_cls[i]))
            if t is not None:
                keep.append(i)
                mapped.append(t)

        if keep:
            det_boxes = det_boxes[keep]
            det_scores = det_scores[keep]
            det_cls = np.array(mapped, dtype=np.int32)
        else:
            det_boxes = np.empty((0, 4), dtype=np.float32)
            det_scores = np.empty(0, dtype=np.float32)
            det_cls = np.empty(0, dtype=np.int32)

        annotated = draw_detections(img, det_boxes, det_cls, det_scores, class_names=names)
        cv2.imwrite(str(out_root / img_path.name), annotated)
        total += 1
        total_dets += len(det_boxes)

    elapsed = time.perf_counter() - t_start

    return {
        "status": "ok",
        "images_processed": total,
        "total_detections": total_dets,
        "output_dir": str(out_root),
        "elapsed_seconds": round(elapsed, 2),
        "avg_fps": round(total / max(elapsed, 1e-6), 1),
    }
