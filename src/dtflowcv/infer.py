from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from dtflowcv.config import write_json
from dtflowcv.deps import blocked_payload, missing_optional_blockers


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
    tracker_class_aware: bool = True,
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
        return blocked_payload([f"invalid_problem_spec:{error}" for error in errors])

    names = class_names(problem)

    blockers = missing_optional_blockers(["cv2", "ultralytics"])
    if blockers:
        return blocked_payload(blockers)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    cmap = model_class_map(getattr(model, "names", {}), names)

    tracker = None
    if enable_tracking:
        from dtflowcv.tracking import SORTTracker
        tracker = SORTTracker(
            max_age=tracker_max_age,
            min_hits=tracker_min_hits,
            iou_threshold=tracker_iou,
            class_aware=tracker_class_aware,
        )

    writer = None
    frame_results: list[dict[str, Any]] = []
    total_detections = 0
    peak_active_tracks = 0
    unique_track_ids: set[int] = set()
    t_start = time.perf_counter()

    try:
        with VideoReader(video_source, sample_fps=sample_fps, max_frames=max_frames) as reader:
            native_fps = reader.fps
            output_fps = sample_fps if sample_fps is not None and sample_fps > 0 else native_fps
            if output_video is not None:
                writer = VideoWriter(output_video, fps=output_fps)

            for frame in reader:
                result = model.predict(frame.image, conf=conf, iou=iou, device=device, verbose=False)[0]

                det_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                det_scores = result.boxes.conf.cpu().numpy().astype(np.float32)
                det_cls = result.boxes.cls.cpu().numpy().astype(np.int32)

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

                tracks = []
                if tracker is not None and len(det_boxes) > 0:
                    tracks = tracker.update(det_boxes, det_scores, det_cls)
                elif tracker is not None:
                    tracks = tracker.update(np.empty((0, 4), dtype=np.float32))
                peak_active_tracks = max(peak_active_tracks, len(tracks))
                unique_track_ids.update(track.track_id for track in tracks)

                detections = [
                    {
                        "class_id": int(class_id),
                        "class_name": names[int(class_id)] if int(class_id) < len(names) else f"cls_{int(class_id)}",
                        "bbox_xyxy": [float(value) for value in bbox],
                        "score": float(score),
                    }
                    for bbox, class_id, score in zip(det_boxes, det_cls, det_scores, strict=False)
                ]

                frame_data: dict[str, Any] = {
                    "frame_index": frame.meta.index,
                    "timestamp_ms": frame.meta.timestamp_ms,
                    "detection_count": len(det_boxes),
                    "detections": detections,
                }
                if tracker is not None:
                    frame_data["active_tracks"] = len(tracks)
                    frame_data["tracks"] = [
                        {
                            "id": t.track_id,
                            "class_id": t.class_id,
                            "class_name": names[t.class_id] if t.class_id < len(names) else f"cls_{t.class_id}",
                            "bbox_xyxy": t.last_bbox.tolist() if t.last_bbox is not None else [],
                            "motion": t.motion.value,
                            "score": round(t.avg_score, 3),
                        }
                        for t in tracks
                    ]
                frame_results.append(frame_data)

                if writer is not None:
                    from dtflowcv.visualize import draw_detections, draw_tracking

                    annotated = frame.image.copy()
                    if tracker is not None and tracks:
                        annotated = draw_tracking(annotated, tracks, class_names=names,
                                                  draw_trajectory=draw_trajectory)
                    elif len(det_boxes) > 0:
                        annotated = draw_detections(annotated, det_boxes, det_cls, det_scores, class_names=names)

                    import cv2
                    elapsed = time.perf_counter() - t_start
                    cur_fps = len(frame_results) / max(elapsed, 1e-6)
                    cv2.putText(annotated, f"FPS: {cur_fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    writer.write(annotated)
    except OSError as exc:
        return {"status": "failed", "errors": [f"video_io_error:{exc}"]}
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - t_start
    frames_processed = len(frame_results)

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
        "peak_active_tracks": peak_active_tracks,
        "unique_track_count": len(unique_track_ids),
        "elapsed_seconds": round(elapsed, 2),
        "avg_fps": round(frames_processed / max(elapsed, 1e-6), 1),
        "native_fps": native_fps,
        "output_video": str(output_video) if output_video else None,
        "output_json": str(output_json) if output_json else None,
        "tracking_enabled": enable_tracking,
        "tracking_class_aware": tracker_class_aware,
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
        return blocked_payload([f"invalid_problem_spec:{error}" for error in errors])

    names = class_names(problem)

    blockers = missing_optional_blockers(["cv2", "ultralytics"])
    if blockers:
        return blocked_payload(blockers)

    from ultralytics import YOLO

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
