from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from dtflowcv.config import load_yaml
from dtflowcv.native import native_status
from dtflowcv.specs import validate_problem_spec
from dtflowcv.train import _dataset_blockers, _module_blockers, _runtime_blockers

CLAIM_BOUNDARY = (
    "Pipeline contracts can be validated locally; model quality is unclaimed until a real benchmark run exists."
)


def project_health(
    problem_path: str | Path = "configs/problem.yaml",
    dataset_path: str | Path | None = None,
    train_config_path: str | Path = "configs/baseline.yolo.yaml",
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    problem_file = Path(problem_path)
    train_config_file = Path(train_config_path)
    problem = _problem_status(problem_file)
    expected_class_count = len(problem["classes"]) if isinstance(problem.get("classes"), list) else None
    report: dict[str, Any] = {
        "status": "ok",
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "problem": problem,
        "training_runtime": _training_runtime_status(train_config_file, dataset_path, expected_class_count),
        "native": native_status(),
        "repo_hygiene": _repo_hygiene(Path(repo_root)),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    blockers = _collect_blockers(report)
    report["build_blockers"] = blockers
    if blockers:
        report["status"] = "blocked"
    return report


def _problem_status(problem_path: Path) -> dict[str, Any]:
    if not problem_path.exists():
        return {"status": "blocked", "path": str(problem_path), "errors": ["problem_spec_missing"]}
    try:
        spec = load_yaml(problem_path)
    except Exception as exc:
        return {"status": "blocked", "path": str(problem_path), "errors": [f"problem_spec_unreadable:{exc}"]}
    errors = validate_problem_spec(spec)
    return {
        "status": "ok" if not errors else "blocked",
        "path": str(problem_path),
        "errors": errors,
        "classes": spec.get("classes", []),
        "claim_boundary": spec.get("project", {}).get("claim_boundary"),
    }


def _training_runtime_status(
    train_config_path: Path,
    dataset_path: str | Path | None,
    expected_class_count: int | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    config: dict[str, Any] = {}
    if train_config_path.exists():
        try:
            config = load_yaml(train_config_path)
        except Exception as exc:
            blockers.append(f"train_config_unreadable:{exc}")
    else:
        blockers.append(f"train_config_missing:{train_config_path}")

    blockers.extend(_module_blockers())
    if config:
        blockers.extend(_runtime_blockers(config))
    if dataset_path is not None:
        blockers.extend(_dataset_blockers(dataset_path, expected_class_count=expected_class_count))

    return {
        "status": "ok" if not blockers else "blocked",
        "train_config": str(train_config_path),
        "dataset": str(dataset_path) if dataset_path is not None else None,
        "dependency_ready": {
            module: importlib.util.find_spec(module) is not None
            for module in ("torch", "ultralytics", "mlflow")
        },
        "blockers": blockers,
    }


def _repo_hygiene(repo_root: Path) -> dict[str, Any]:
    tracked_large_files = _tracked_large_files(repo_root)
    return {
        "tracked_large_file_threshold_mb": 100,
        "tracked_large_files": tracked_large_files,
        "push_risk": "high" if tracked_large_files else "low",
    }


def _tracked_large_files(repo_root: Path, threshold_bytes: int = 100 * 1024 * 1024) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    paths = [Path(raw.decode()) for raw in result.stdout.split(b"\0") if raw]
    large_files: list[dict[str, Any]] = []
    for relative_path in paths:
        full_path = repo_root / relative_path
        try:
            size = full_path.stat().st_size
        except OSError:
            continue
        if size >= threshold_bytes:
            large_files.append({"path": str(relative_path), "size_bytes": size})
    return sorted(large_files, key=lambda item: int(item["size_bytes"]), reverse=True)


def _collect_blockers(report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    blockers.extend(f"problem:{error}" for error in report["problem"].get("errors", []))
    blockers.extend(f"training:{blocker}" for blocker in report["training_runtime"].get("blockers", []))
    for large_file in report["repo_hygiene"]["tracked_large_files"]:
        blockers.append(f"tracked_large_file:{large_file['path']}")
    return blockers
