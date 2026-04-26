"""Model training, prediction, export, card, and registry boundaries."""

from dtflowcv.models.backend import DetectionBackend, DetectionResult
from dtflowcv.models.cards import ModelCard, write_model_card
from dtflowcv.models.downloader import stage_model_artifact
from dtflowcv.models.export import export_engine, export_onnx, export_torchscript
from dtflowcv.models.export_validate import validate_export
from dtflowcv.models.onnxruntime_backend import OnnxRuntimeDetectionBackend
from dtflowcv.models.predict import predict_ultralytics_yolo
from dtflowcv.models.registry import ModelRegistry, ModelRegistryEntry, build_registry_entry
from dtflowcv.models.train import train_yolo_baseline
from dtflowcv.models.ultralytics_backend import UltralyticsDetectionBackend

__all__ = [
    "DetectionBackend",
    "DetectionResult",
    "ModelCard",
    "ModelRegistry",
    "ModelRegistryEntry",
    "OnnxRuntimeDetectionBackend",
    "UltralyticsDetectionBackend",
    "export_engine",
    "export_onnx",
    "export_torchscript",
    "predict_ultralytics_yolo",
    "build_registry_entry",
    "stage_model_artifact",
    "train_yolo_baseline",
    "validate_export",
    "write_model_card",
]
