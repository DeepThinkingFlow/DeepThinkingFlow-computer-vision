from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dtflowcv.config import write_json

MODEL_STATUSES = {"candidate", "validated", "rejected", "staging", "deployed", "retired"}


@dataclass
class ModelRegistryEntry:
    name: str
    version: str
    task: str
    format: str
    path: str
    sha256: str
    license: str
    source: str
    status: str = "candidate"
    model_card: str = ""
    benchmark_report: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "format": self.format,
            "path": self.path,
            "sha256": self.sha256,
            "license": self.license,
            "source": self.source,
            "status": self.status,
            "model_card": self.model_card,
            "benchmark_report": self.benchmark_report,
            "metadata": self.metadata,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelRegistryEntry:
        return cls(
            name=str(payload.get("name", "")),
            version=str(payload.get("version", "")),
            task=str(payload.get("task", "object_detection")),
            format=str(payload.get("format", "")),
            path=str(payload.get("path", "")),
            sha256=str(payload.get("sha256", "")),
            license=str(payload.get("license", "")),
            source=str(payload.get("source", "")),
            status=str(payload.get("status", "candidate")),
            model_card=str(payload.get("model_card", "")),
            benchmark_report=str(payload.get("benchmark_report", "")),
            metadata=dict(payload.get("metadata", {})),
            created=str(payload.get("created", "")),
        )


class ModelRegistry:
    def __init__(self, registry_path: str | Path = "artifacts/model_registry.json") -> None:
        self.path = Path(registry_path)
        self.entries: list[ModelRegistryEntry] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries = [ModelRegistryEntry.from_dict(item) for item in payload.get("models", [])]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            self.path,
            {
                "registry_version": "1.0",
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "models": [entry.to_dict() for entry in self.entries],
            },
        )

    def register(self, entry: ModelRegistryEntry, *, replace: bool = False) -> dict[str, Any]:
        errors = validate_model_registry_entry(entry)
        if errors:
            return {"status": "failed", "errors": errors}
        existing = self.get(entry.name, entry.version)
        if existing is not None and not replace:
            return {"status": "failed", "errors": [f"model_already_registered:{entry.name}:{entry.version}"]}
        if not entry.created:
            entry.created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if existing is not None:
            self.entries = [candidate for candidate in self.entries if not _same_model(candidate, entry)]
        self.entries.append(entry)
        self.save()
        return {"status": "ok", "registry": str(self.path), "model": entry.to_dict()}

    def get(self, name: str, version: str | None = None) -> ModelRegistryEntry | None:
        for entry in reversed(self.entries):
            if entry.name == name and (version is None or entry.version == version):
                return entry
        return None

    def list_models(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.entries]

    def promote(self, name: str, version: str, status: str) -> dict[str, Any]:
        if status not in MODEL_STATUSES:
            return {"status": "failed", "errors": [f"invalid_model_status:{status}"]}
        entry = self.get(name, version)
        if entry is None:
            return {"status": "failed", "errors": [f"model_not_found:{name}:{version}"]}
        entry.status = status
        self.save()
        return {"status": "ok", "registry": str(self.path), "model": entry.to_dict()}


def build_registry_entry(
    *,
    name: str,
    version: str,
    path: str | Path,
    license_id: str,
    source: str,
    task: str = "object_detection",
    format: str | None = None,
    status: str = "candidate",
    model_card: str | Path | None = None,
    benchmark_report: str | Path | None = None,
) -> ModelRegistryEntry:
    model_path = Path(path)
    model_format = format or _format_from_path(model_path)
    return ModelRegistryEntry(
        name=name,
        version=version,
        task=task,
        format=model_format,
        path=str(model_path),
        sha256=file_sha256(model_path) if model_path.exists() else "",
        license=license_id,
        source=source,
        status=status,
        model_card=str(model_card or ""),
        benchmark_report=str(benchmark_report or ""),
    )


def validate_model_registry_entry(entry: ModelRegistryEntry) -> list[str]:
    errors: list[str] = []
    if not entry.name:
        errors.append("missing_model_name")
    if not entry.version:
        errors.append("missing_model_version")
    if entry.status not in MODEL_STATUSES:
        errors.append(f"invalid_model_status:{entry.status}")
    if not entry.license:
        errors.append("missing_model_license")
    path = Path(entry.path)
    if not path.exists():
        errors.append(f"missing_model_artifact:{entry.path}")
    elif file_sha256(path) != entry.sha256:
        errors.append(f"model_sha256_mismatch:{entry.path}")
    return errors


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _format_from_path(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "pt":
        return "pytorch"
    if suffix == "engine":
        return "tensorrt"
    return suffix or "unknown"


def _same_model(left: ModelRegistryEntry, right: ModelRegistryEntry) -> bool:
    return left.name == right.name and left.version == right.version


__all__ = [
    "MODEL_STATUSES",
    "ModelRegistry",
    "ModelRegistryEntry",
    "build_registry_entry",
    "file_sha256",
    "validate_model_registry_entry",
]
