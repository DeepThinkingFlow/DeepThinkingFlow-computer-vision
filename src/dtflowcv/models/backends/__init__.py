from __future__ import annotations

from dtflowcv.models.backends.onnxruntime import OnnxRuntimeDetectionBackend
from dtflowcv.models.backends.torchscript import TorchScriptDetectionBackend
from dtflowcv.models.backends.ultralytics import UltralyticsDetectionBackend

__all__ = [
    "OnnxRuntimeDetectionBackend",
    "TorchScriptDetectionBackend",
    "UltralyticsDetectionBackend",
]
