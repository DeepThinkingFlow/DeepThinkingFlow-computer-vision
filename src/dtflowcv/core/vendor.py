from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

PROJECT_LICENSE = "Apache-2.0"

ALLOWED_VENDOR_LICENSES = {
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Zlib",
    "0BSD",
    "CC0-1.0",
}
REVIEW_VENDOR_LICENSES = {"MPL-2.0", "LGPL-2.1", "LGPL-3.0", "EPL-2.0", "CDDL"}
FORBIDDEN_VENDOR_LICENSES = {
    "GPL-2.0",
    "GPL-3.0",
    "AGPL-3.0",
    "SSPL",
    "unknown",
    "custom-non-commercial",
    "non-commercial",
    "research-only",
    "no-license",
}
REQUIRED_ROOT_FILES = [
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "LICENSE_POLICY.md",
    "VENDOR_POLICY.md",
    "third_party/MANIFEST.json",
]
REQUIRED_VENDOR_FILES = ["LICENSE", "UPSTREAM.md", "PATCHES.md"]
MODEL_ARTIFACT_PATTERNS = ("*.pt", "*.pth", "*.onnx", "*.engine", "*.safetensors", "*.torchscript")
FORBIDDEN_LICENSE_MARKERS = [
    "agpl",
    "gnu affero",
    "gnu general public license",
    "sspl",
    "non-commercial",
    "research only",
]


def vendor_check(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root)
    manifest_path = root / "third_party" / "MANIFEST.json"
    errors: list[str] = []
    warnings: list[str] = []
    forbidden_components = 0

    manifest = _read_manifest(manifest_path, errors)
    components = manifest.get("components", []) if isinstance(manifest, dict) else []
    if not isinstance(components, list):
        errors.append("vendor_manifest_schema:components_must_be_list")
        components = []

    manifest_paths: set[str] = set()
    component_summaries: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            errors.append(f"vendor_manifest_schema:component_{index}_must_be_object")
            continue
        summary, component_forbidden = _validate_component(root, index, component, errors, warnings)
        forbidden_components += component_forbidden
        component_summaries.append(summary)
        vendored_path = component.get("vendored_path")
        if isinstance(vendored_path, str):
            manifest_paths.add(_normalize_relpath(vendored_path))

    _find_unmanifested_vendor_dirs(root, manifest_paths, errors)
    _scan_permissive_license_files(root, errors)

    return {
        "status": "failed" if errors else "ok",
        "manifest": str(manifest_path),
        "vendored_components": len(components),
        "forbidden_components": forbidden_components,
        "components": component_summaries,
        "errors": errors,
        "warnings": warnings,
    }


def license_check(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root)
    errors: list[str] = []
    warnings: list[str] = []

    for required in REQUIRED_ROOT_FILES:
        if not (root / required).exists():
            errors.append(f"missing_required_license_file:{required}")

    detected_license = _detect_project_license(root / "LICENSE")
    if detected_license != PROJECT_LICENSE:
        errors.append(f"project_license_mismatch:expected:{PROJECT_LICENSE}:actual:{detected_license}")

    vendor_result = vendor_check(root)
    errors.extend(f"vendor_check:{item}" for item in vendor_result["errors"])
    warnings.extend(f"vendor_check:{item}" for item in vendor_result["warnings"])

    tracked_model_artifacts = _tracked_model_artifacts(root)
    if tracked_model_artifacts:
        errors.extend(f"tracked_model_artifact:{path}" for path in tracked_model_artifacts)

    warnings.append(
        "optional_dependency:ultralytics may carry AGPL-3.0 obligations unless covered by an enterprise license"
    )

    return {
        "status": "failed" if errors else "ok",
        "project_license": detected_license,
        "expected_project_license": PROJECT_LICENSE,
        "vendored_components": vendor_result["vendored_components"],
        "forbidden_components": vendor_result["forbidden_components"],
        "tracked_model_artifacts": tracked_model_artifacts,
        "warnings": warnings,
        "errors": errors,
    }


def _read_manifest(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append("missing_vendor_manifest:third_party/MANIFEST.json")
        return {"components": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"vendor_manifest_invalid_json:{exc.msg}")
        return {"components": []}
    if not isinstance(payload, dict):
        errors.append("vendor_manifest_schema:root_must_be_object")
        return {"components": []}
    return payload


def _validate_component(
    root: Path,
    index: int,
    component: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], int]:
    required_fields = [
        "name",
        "upstream_url",
        "upstream_commit",
        "license",
        "vendored_path",
        "reason",
        "local_modifications",
        "patches_file",
        "owner",
        "allowed_in_core",
    ]
    name = str(component.get("name", f"component_{index}"))
    for field in required_fields:
        if field not in component:
            errors.append(f"vendor_component:{name}:missing_field:{field}")

    license_id = str(component.get("license", "unknown"))
    vendored_path_raw = str(component.get("vendored_path", ""))
    vendored_path = root / vendored_path_raw
    forbidden = 0

    if license_id in FORBIDDEN_VENDOR_LICENSES:
        forbidden += 1
        errors.append(f"vendor_component:{name}:forbidden_license:{license_id}")
    elif license_id in REVIEW_VENDOR_LICENSES:
        warnings.append(f"vendor_component:{name}:review_required_license:{license_id}")
    elif license_id not in ALLOWED_VENDOR_LICENSES:
        forbidden += 1
        errors.append(f"vendor_component:{name}:unknown_license:{license_id}")

    if "permissive" in vendored_path.parts and license_id not in ALLOWED_VENDOR_LICENSES:
        errors.append(f"vendor_component:{name}:non_permissive_license_in_permissive_zone:{license_id}")

    if not vendored_path_raw:
        errors.append(f"vendor_component:{name}:missing_vendored_path")
    elif not vendored_path.exists():
        errors.append(f"vendor_component:{name}:missing_vendored_path:{vendored_path_raw}")
    else:
        for required in REQUIRED_VENDOR_FILES:
            if not (vendored_path / required).exists():
                errors.append(f"vendor_component:{name}:missing_file:{required}")
        patches_file = component.get("patches_file")
        if isinstance(patches_file, str) and not (root / patches_file).exists():
            errors.append(f"vendor_component:{name}:missing_patches_file:{patches_file}")

    return {
        "name": name,
        "license": license_id,
        "vendored_path": vendored_path_raw,
        "allowed_in_core": bool(component.get("allowed_in_core", False)),
    }, forbidden


def _find_unmanifested_vendor_dirs(root: Path, manifest_paths: set[str], errors: list[str]) -> None:
    third_party = root / "third_party"
    for zone in ("permissive", "agpl"):
        zone_path = third_party / zone
        if not zone_path.exists():
            continue
        for child in zone_path.iterdir():
            if child.name.startswith(".") or not child.is_dir():
                continue
            rel = _normalize_relpath(child.relative_to(root))
            if rel not in manifest_paths:
                errors.append(f"unmanifested_vendor_component:{rel}")


def _scan_permissive_license_files(root: Path, errors: list[str]) -> None:
    permissive = root / "third_party" / "permissive"
    if not permissive.exists():
        return
    for path in permissive.rglob("*"):
        if not path.is_file() or not path.name.upper().startswith("LICENSE"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for marker in FORBIDDEN_LICENSE_MARKERS:
            if marker in text:
                errors.append(f"forbidden_license_marker_in_permissive_zone:{path.relative_to(root)}:{marker}")
                break


def _detect_project_license(path: Path) -> str:
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="replace")
    if "Apache License" in text and "Version 2.0" in text:
        return PROJECT_LICENSE
    return "unknown"


def _tracked_model_artifacts(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", *MODEL_ARTIFACT_PATTERNS],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _normalize_relpath(path: str | Path) -> str:
    return Path(path).as_posix().strip("/")


__all__ = [
    "ALLOWED_VENDOR_LICENSES",
    "FORBIDDEN_VENDOR_LICENSES",
    "PROJECT_LICENSE",
    "REVIEW_VENDOR_LICENSES",
    "license_check",
    "vendor_check",
]
