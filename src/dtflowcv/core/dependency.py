from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import sys
from typing import Any

from dtflowcv.deps import missing_optional_blockers

CORE_MODULES = ["numpy", "PIL", "yaml", "typer"]
EXTRA_MODULES = {
    "train": ["mlflow", "torch", "ultralytics"],
    "export": ["onnx", "onnxruntime", "torch", "ultralytics"],
    "deploy": ["onnxruntime", "cv2"],
    "video": ["cv2"],
    "viz": ["cv2", "matplotlib"],
    "analysis": ["cv2", "matplotlib", "pandas"],
    "native": ["maturin"],
}


def dependency_check() -> dict[str, Any]:
    core_missing = [module for module in CORE_MODULES if importlib.util.find_spec(module) is None]
    extras: dict[str, dict[str, Any]] = {}
    for extra, modules in EXTRA_MODULES.items():
        blockers = missing_optional_blockers(modules)
        extras[extra] = {
            "status": "available" if not blockers else "missing",
            "build_blockers": blockers,
        }

    package_versions = {
        name: _version(name)
        for name in [
            "numpy",
            "Pillow",
            "PyYAML",
            "typer",
            "opencv-python",
            "onnxruntime",
            "torch",
            "ultralytics",
            "mlflow",
        ]
    }

    license_warnings = []
    if importlib.util.find_spec("ultralytics") is not None:
        license_warnings.append("ultralytics:AGPL-3.0 by default unless covered by enterprise license")

    return {
        "status": "failed" if core_missing else "ok",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "core_missing": core_missing,
        "extras": extras,
        "package_versions": package_versions,
        "licenses": {
            "forbidden": [],
            "warnings": license_warnings,
        },
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


__all__ = ["dependency_check"]
