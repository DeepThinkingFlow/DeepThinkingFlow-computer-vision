from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class DetectionResult:
    class_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    class_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox_xyxy": list(self.bbox_xyxy),
            "score": self.score,
        }


class DetectionBackend(Protocol):
    def load(self) -> dict[str, Any]: ...

    def predict_image(self, image: str | Path, **kwargs: Any) -> list[dict[str, Any]] | dict[str, Any]: ...

    def predict_batch(self, images: list[str | Path], **kwargs: Any) -> list[Any] | dict[str, Any]: ...

    def backend_info(self) -> dict[str, Any]: ...


__all__ = ["DetectionBackend", "DetectionResult"]
