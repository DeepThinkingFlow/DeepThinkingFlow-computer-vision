from __future__ import annotations

import importlib.util
from typing import Any

OPTIONAL_DEPENDENCIES: dict[str, tuple[str, str]] = {
    "cv2": ("opencv-python", "python -m pip install -e '.[viz,video]'"),
    "matplotlib": ("matplotlib", "python -m pip install -e '.[viz]'"),
    "mlflow": ("mlflow", "python -m pip install -e '.[train]'"),
    "onnx": ("onnx", "python -m pip install -e '.[export]'"),
    "onnxruntime": ("onnxruntime", "python -m pip install -e '.[deploy]'"),
    "torch": ("torch", "python -m pip install -e '.[train]'"),
    "ultralytics": ("ultralytics", "python -m pip install -e '.[train]'"),
}


def missing_optional_blockers(modules: list[str]) -> list[str]:
    blockers: list[str] = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            package, install_hint = OPTIONAL_DEPENDENCIES.get(module, (module, "install the required extra"))
            blockers.append(f"missing_python_module:{package}: install with {install_hint}")
    return blockers


def blocked_payload(blockers: list[str], **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "blocked", "build_blockers": blockers}
    payload.update(extra)
    return payload
