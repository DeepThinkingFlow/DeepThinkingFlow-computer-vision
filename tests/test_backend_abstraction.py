from __future__ import annotations


def test_backend_protocol_exports_expected_methods() -> None:
    from dtflowcv.models.backend import DetectionResult
    from dtflowcv.models.backends.onnxruntime import OnnxRuntimeDetectionBackend as NestedOnnxRuntimeBackend
    from dtflowcv.models.backends.torchscript import TorchScriptDetectionBackend
    from dtflowcv.models.backends.ultralytics import UltralyticsDetectionBackend as NestedUltralyticsBackend
    from dtflowcv.models.onnxruntime_backend import OnnxRuntimeDetectionBackend
    from dtflowcv.models.ultralytics_backend import UltralyticsDetectionBackend

    detection = DetectionResult(class_id=1, class_name="car", bbox_xyxy=(0.0, 1.0, 2.0, 3.0), score=0.9)

    assert detection.to_dict()["bbox_xyxy"] == [0.0, 1.0, 2.0, 3.0]
    assert hasattr(UltralyticsDetectionBackend, "predict_image")
    assert hasattr(UltralyticsDetectionBackend, "predict_batch")
    assert hasattr(UltralyticsDetectionBackend, "backend_info")
    assert hasattr(UltralyticsDetectionBackend, "load")
    assert hasattr(OnnxRuntimeDetectionBackend, "predict_image")
    assert hasattr(OnnxRuntimeDetectionBackend, "predict_batch")
    assert hasattr(OnnxRuntimeDetectionBackend, "backend_info")
    assert hasattr(OnnxRuntimeDetectionBackend, "load")
    assert NestedUltralyticsBackend is UltralyticsDetectionBackend
    assert NestedOnnxRuntimeBackend is OnnxRuntimeDetectionBackend
    assert hasattr(TorchScriptDetectionBackend, "load")
