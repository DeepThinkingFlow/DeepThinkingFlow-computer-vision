from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dtflowcv.deps import blocked_payload, missing_optional_blockers


class OnnxRuntimeDetectionBackend:
    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int = 640,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.providers = providers
        self._blockers = missing_optional_blockers(["onnxruntime"])
        self._session: Any | None = None

    def load(self) -> dict[str, Any]:
        if self._blockers:
            return blocked_payload(self._blockers, backend="onnxruntime")
        if not self.model_path.exists():
            return {"status": "failed", "backend": "onnxruntime", "errors": [f"missing_model:{self.model_path}"]}
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        return {"status": "ok", "backend": "onnxruntime", "model": str(self.model_path)}

    def backend_info(self) -> dict[str, Any]:
        if self._blockers:
            return blocked_payload(self._blockers, backend="onnxruntime")
        return {
            "status": "ok",
            "backend": "onnxruntime",
            "model": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "input_size": self.input_size,
            "providers": self._session.get_providers() if self._session is not None else self.providers,
            "claim_boundary": "Raw ONNXRuntime session support; detection postprocess parity is not claimed here.",
        }

    def predict_image(self, image: str | Path, **_: Any) -> list[dict[str, Any]] | dict[str, Any]:
        if self._blockers:
            return blocked_payload(self._blockers, backend="onnxruntime")
        load_result = self.load()
        if load_result.get("status") != "ok":
            return load_result
        assert self._session is not None
        array = _load_image_nchw(image, self.input_size)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: array})
        return {
            "status": "ok",
            "backend": "onnxruntime",
            "raw_output_shapes": [list(output.shape) for output in outputs],
            "detections": [],
            "claim_boundary": "Raw output only; NMS/class/bbox postprocess must be added before deployment claims.",
        }

    def predict_batch(self, images: list[str | Path], **kwargs: Any) -> list[dict[str, Any]] | dict[str, Any]:
        if self._blockers:
            return blocked_payload(self._blockers, backend="onnxruntime")
        return [self.predict_image(image, **kwargs) for image in images]


def _load_image_nchw(path: str | Path, input_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((input_size, input_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, :, :, :]


__all__ = ["OnnxRuntimeDetectionBackend"]
