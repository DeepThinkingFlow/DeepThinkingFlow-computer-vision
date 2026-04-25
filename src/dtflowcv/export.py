from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dtflowcv.config import write_json


@dataclass
class ModelCard:
    """Model metadata for versioning and deployment."""
    name: str
    architecture: str = ""
    input_size: tuple[int, int] = (640, 640)
    classes: list[str] = field(default_factory=list)
    export_format: str = ""
    source_checkpoint: str = ""
    source_checkpoint_hash: str = ""
    training_config_hash: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    created: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "input_size": list(self.input_size),
            "classes": self.classes,
            "export_format": self.export_format,
            "source_checkpoint": self.source_checkpoint,
            "source_checkpoint_hash": self.source_checkpoint_hash,
            "training_config_hash": self.training_config_hash,
            "metrics": self.metrics,
            "created": self.created,
            "notes": self.notes,
        }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def export_onnx(
    model_path: str | Path,
    output_path: str | Path,
    *,
    input_size: int = 640,
    dynamic_batch: bool = True,
    simplify: bool = True,
    opset: int = 17,
    half: bool = False,
) -> dict[str, Any]:
    """Export Ultralytics YOLO model to ONNX.

    Args:
        model_path: Path to .pt checkpoint.
        output_path: Output .onnx path.
        input_size: Input image size.
        dynamic_batch: Enable dynamic batch dimension.
        simplify: Simplify ONNX graph.
        opset: ONNX opset version.
        half: Export in FP16.

    Returns:
        Summary dict with paths, hashes, status.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"status": "blocked", "reason": "ultralytics not installed"}

    model_p = Path(model_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_p))
    export_result = model.export(
        format="onnx",
        imgsz=input_size,
        dynamic=dynamic_batch,
        simplify=simplify,
        opset=opset,
        half=half,
    )

    # Ultralytics returns the export path
    exported_path = Path(str(export_result))
    if exported_path.exists() and exported_path != out_p:
        import shutil
        shutil.move(str(exported_path), str(out_p))

    if not out_p.exists():
        return {"status": "failed", "reason": f"ONNX file not found at {out_p}"}

    return {
        "status": "ok",
        "format": "onnx",
        "source": str(model_p),
        "output": str(out_p),
        "source_sha256": _file_sha256(model_p),
        "output_sha256": _file_sha256(out_p),
        "output_size_mb": out_p.stat().st_size / (1024 * 1024),
        "input_size": input_size,
        "opset": opset,
        "dynamic_batch": dynamic_batch,
        "half": half,
    }


def export_torchscript(
    model_path: str | Path,
    output_path: str | Path,
    *,
    input_size: int = 640,
    half: bool = False,
) -> dict[str, Any]:
    """Export Ultralytics YOLO model to TorchScript."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"status": "blocked", "reason": "ultralytics not installed"}

    model_p = Path(model_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_p))
    export_result = model.export(
        format="torchscript",
        imgsz=input_size,
        half=half,
    )

    exported_path = Path(str(export_result))
    if exported_path.exists() and exported_path != out_p:
        import shutil
        shutil.move(str(exported_path), str(out_p))

    if not out_p.exists():
        return {"status": "failed", "reason": f"TorchScript file not found at {out_p}"}

    return {
        "status": "ok",
        "format": "torchscript",
        "source": str(model_p),
        "output": str(out_p),
        "source_sha256": _file_sha256(model_p),
        "output_sha256": _file_sha256(out_p),
        "output_size_mb": out_p.stat().st_size / (1024 * 1024),
        "input_size": input_size,
        "half": half,
    }


def export_engine(
    model_path: str | Path,
    output_path: str | Path,
    *,
    input_size: int = 640,
    half: bool = True,
    workspace: int = 4,
) -> dict[str, Any]:
    """Export YOLO model to TensorRT engine (requires TensorRT + GPU)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"status": "blocked", "reason": "ultralytics not installed"}

    try:
        import torch
        if not torch.cuda.is_available():
            return {"status": "blocked", "reason": "CUDA not available for TensorRT export"}
    except ImportError:
        return {"status": "blocked", "reason": "torch not installed"}

    model_p = Path(model_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_p))
    export_result = model.export(
        format="engine",
        imgsz=input_size,
        half=half,
        workspace=workspace,
    )

    exported_path = Path(str(export_result))
    if exported_path.exists() and exported_path != out_p:
        import shutil
        shutil.move(str(exported_path), str(out_p))

    if not out_p.exists():
        return {"status": "failed", "reason": f"TensorRT engine not found at {out_p}"}

    return {
        "status": "ok",
        "format": "tensorrt",
        "source": str(model_p),
        "output": str(out_p),
        "source_sha256": _file_sha256(model_p),
        "output_sha256": _file_sha256(out_p),
        "output_size_mb": out_p.stat().st_size / (1024 * 1024),
        "input_size": input_size,
        "half": half,
        "workspace_gb": workspace,
    }


def validate_export(
    original_model: str | Path,
    exported_model: str | Path,
    test_image: str | Path,
    *,
    max_diff: float = 1e-3,
) -> dict[str, Any]:
    """Validate exported model by comparing outputs with original.

    Runs same input through both models, asserts max absolute difference.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"status": "blocked", "reason": "ultralytics not installed"}

    original = YOLO(str(original_model))
    exported = YOLO(str(exported_model))

    orig_result = original.predict(str(test_image), verbose=False)[0]
    exp_result = exported.predict(str(test_image), verbose=False)[0]

    import numpy as np

    orig_boxes = orig_result.boxes.xyxy.cpu().numpy()
    exp_boxes = exp_result.boxes.xyxy.cpu().numpy()

    if len(orig_boxes) == 0 and len(exp_boxes) == 0:
        return {"status": "ok", "max_diff": 0.0, "detections_match": True, "note": "both empty"}

    if len(orig_boxes) != len(exp_boxes):
        return {
            "status": "warning",
            "max_diff": float("inf"),
            "detections_match": False,
            "original_count": len(orig_boxes),
            "exported_count": len(exp_boxes),
            "note": "different detection counts",
        }

    max_box_diff = float(np.max(np.abs(orig_boxes - exp_boxes)))
    passed = max_box_diff <= max_diff

    return {
        "status": "ok" if passed else "failed",
        "max_diff": max_box_diff,
        "threshold": max_diff,
        "detections_match": True,
        "detection_count": len(orig_boxes),
    }


def write_model_card(
    card: ModelCard,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write model card as JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "model_card.json"
    write_json(json_path, card.to_dict())
    return {"json": str(json_path)}
