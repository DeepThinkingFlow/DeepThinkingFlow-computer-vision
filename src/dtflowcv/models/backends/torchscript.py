from __future__ import annotations

from pathlib import Path
from typing import Any

from dtflowcv.deps import blocked_payload, missing_optional_blockers


class TorchScriptDetectionBackend:
    def __init__(self, model_path: str | Path, *, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._blockers = missing_optional_blockers(["torch"])
        self._model: Any | None = None

    def load(self) -> dict[str, Any]:
        if self._blockers:
            return blocked_payload(self._blockers, backend="torchscript")
        if not self.model_path.exists():
            return {"status": "failed", "backend": "torchscript", "errors": [f"missing_model:{self.model_path}"]}
        if self._model is None:
            import torch

            self._model = torch.jit.load(str(self.model_path), map_location=self.device)
            self._model.eval()
        return {"status": "ok", "backend": "torchscript", "model": str(self.model_path)}

    def backend_info(self) -> dict[str, Any]:
        if self._blockers:
            return blocked_payload(self._blockers, backend="torchscript")
        return {
            "status": "ok",
            "backend": "torchscript",
            "model": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "device": self.device,
            "claim_boundary": "TorchScript load path only; detection postprocess parity is not claimed here.",
        }

    def predict_image(self, image: str | Path, **_: Any) -> list[dict[str, Any]] | dict[str, Any]:
        load_result = self.load()
        if load_result.get("status") != "ok":
            return load_result
        return {
            "status": "blocked",
            "backend": "torchscript",
            "build_blockers": ["torchscript_postprocess_not_implemented"],
            "image": str(image),
        }

    def predict_batch(self, images: list[str | Path], **kwargs: Any) -> list[Any] | dict[str, Any]:
        if self._blockers:
            return blocked_payload(self._blockers, backend="torchscript")
        return [self.predict_image(image, **kwargs) for image in images]


__all__ = ["TorchScriptDetectionBackend"]
