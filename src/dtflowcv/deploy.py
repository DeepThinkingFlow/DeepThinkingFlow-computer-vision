from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dtflowcv.config import write_json
from dtflowcv.deps import missing_optional_blockers


@dataclass
class ModelRegistryEntry:
    """A single model entry in the registry."""
    name: str
    version: str
    architecture: str
    checkpoint_path: str
    checkpoint_hash: str = ""
    export_artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    input_size: tuple[int, int] = (640, 640)
    classes: list[str] = field(default_factory=list)
    training_config: str = ""
    created: str = ""
    status: str = "candidate"  # candidate | validated | deployed | retired

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "architecture": self.architecture,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "export_artifacts": self.export_artifacts,
            "metrics": self.metrics,
            "input_size": list(self.input_size),
            "classes": self.classes,
            "training_config": self.training_config,
            "created": self.created,
            "status": self.status,
        }


class ModelRegistry:
    """Local model registry for versioning and tracking deployments.

    Stores model metadata as JSON, supports add/list/promote/retire.
    """

    def __init__(self, registry_path: str | Path = "artifacts/model_registry.json") -> None:
        self._path = Path(registry_path)
        self._entries: list[ModelRegistryEntry] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._entries = []
        for item in data.get("models", []):
            entry = ModelRegistryEntry(
                name=item.get("name", ""),
                version=item.get("version", ""),
                architecture=item.get("architecture", ""),
                checkpoint_path=item.get("checkpoint_path", ""),
                checkpoint_hash=item.get("checkpoint_hash", ""),
                export_artifacts=item.get("export_artifacts", {}),
                metrics=item.get("metrics", {}),
                input_size=tuple(item.get("input_size", [640, 640])),
                classes=item.get("classes", []),
                training_config=item.get("training_config", ""),
                created=item.get("created", ""),
                status=item.get("status", "candidate"),
            )
            self._entries.append(entry)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "registry_version": "1.0",
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": [e.to_dict() for e in self._entries],
        }
        write_json(self._path, data)

    def add(self, entry: ModelRegistryEntry) -> None:
        if not entry.created:
            entry.created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not entry.checkpoint_hash and Path(entry.checkpoint_path).exists():
            h = hashlib.sha256()
            h.update(Path(entry.checkpoint_path).read_bytes())
            entry.checkpoint_hash = h.hexdigest()
        self._entries.append(entry)
        self.save()

    def list_models(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def get(self, name: str, version: str | None = None) -> ModelRegistryEntry | None:
        for e in reversed(self._entries):
            if e.name == name and (version is None or e.version == version):
                return e
        return None

    def promote(self, name: str, version: str, new_status: str = "validated") -> bool:
        entry = self.get(name, version)
        if entry is None:
            return False
        entry.status = new_status
        self.save()
        return True

    @property
    def deployed_model(self) -> ModelRegistryEntry | None:
        for e in reversed(self._entries):
            if e.status == "deployed":
                return e
        return None


def environment_check() -> dict[str, Any]:
    """Check runtime environment for deployment readiness.

    Verifies Python version, key packages, GPU availability, disk space.
    """
    import platform
    import sys

    checks: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    # Key packages
    packages = {
        "numpy": "numpy",
        "opencv": "cv2",
        "pillow": "PIL",
        "pyyaml": "yaml",
        "typer": "typer",
        "ultralytics": "ultralytics",
        "torch": "torch",
        "mlflow": "mlflow",
        "onnxruntime": "onnxruntime",
    }

    pkg_status: dict[str, str] = {}
    for name, module in packages.items():
        spec = importlib.util.find_spec(module)
        if spec is not None:
            try:
                mod = __import__(module)
                ver = getattr(mod, "__version__", "installed")
                pkg_status[name] = str(ver)
            except Exception:
                pkg_status[name] = "found_but_import_error"
        else:
            pkg_status[name] = "not_installed"
    checks["packages"] = pkg_status

    # GPU
    try:
        import torch
        checks["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            checks["cuda_device"] = torch.cuda.get_device_name(0)
            checks["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
    except ImportError:
        checks["cuda_available"] = False

    # Disk space
    import shutil
    usage = shutil.disk_usage(Path.cwd())
    checks["disk_free_gb"] = round(usage.free / (1024**3), 1)
    checks["disk_total_gb"] = round(usage.total / (1024**3), 1)

    # Memory
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    checks["ram_total_mb"] = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    checks["ram_available_mb"] = int(line.split()[1]) // 1024
    except OSError:
        pass

    # Overall readiness
    blockers = []
    if pkg_status.get("numpy") == "not_installed":
        blockers.append("missing_python_module:numpy: install with python -m pip install -e '.[dev]'")
    blockers.extend(missing_optional_blockers(["cv2", "ultralytics", "onnxruntime"]))

    checks["build_blockers"] = blockers
    checks["blockers"] = blockers
    checks["ready"] = len(blockers) == 0
    checks["status"] = "ok" if checks["ready"] else "blocked"

    return checks
