from __future__ import annotations

from pathlib import Path
from typing import Any

from dtflowcv.deps import blocked_payload, missing_optional_blockers
from dtflowcv.models.backend import DetectionResult
from dtflowcv.predict import model_class_map, predict_ultralytics_yolo


class UltralyticsDetectionBackend:
    def __init__(
        self,
        model_path: str | Path,
        *,
        class_names: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names = class_names or []
        self.device = device
        self._blockers = missing_optional_blockers(["ultralytics"])
        self._model: Any | None = None

    def load(self) -> dict[str, Any]:
        if self._blockers:
            return blocked_payload(self._blockers, backend="ultralytics")
        if not self.model_path.exists():
            return {"status": "failed", "backend": "ultralytics", "errors": [f"missing_model:{self.model_path}"]}
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return {"status": "ok", "backend": "ultralytics", "model": str(self.model_path)}

    def backend_info(self) -> dict[str, Any]:
        if self._blockers:
            return blocked_payload(self._blockers, backend="ultralytics")
        return {
            "status": "ok",
            "backend": "ultralytics",
            "model": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "license_boundary": "Ultralytics may carry AGPL-3.0 obligations unless covered by enterprise license.",
        }

    def predict_image(
        self,
        image: str | Path,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        **_: Any,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        if self._blockers:
            return blocked_payload(self._blockers, backend="ultralytics")
        load_result = self.load()
        if load_result.get("status") != "ok":
            return load_result
        assert self._model is not None
        result = self._model.predict(
            source=str(image),
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        for box, class_id, score in zip(xyxy, cls, scores, strict=True):
            class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else None
            detections.append(
                DetectionResult(
                    class_id=int(class_id),
                    class_name=class_name,
                    bbox_xyxy=tuple(float(value) for value in box),
                    score=float(score),
                ).to_dict()
            )
        return detections

    def predict_batch(self, images: list[str | Path], **kwargs: Any) -> list[Any] | dict[str, Any]:
        if self._blockers:
            return blocked_payload(self._blockers, backend="ultralytics")
        return [self.predict_image(image, **kwargs) for image in images]


__all__ = [
    "UltralyticsDetectionBackend",
    "model_class_map",
    "predict_ultralytics_yolo",
]
