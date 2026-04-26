from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dtflowcv.deps import blocked_payload
from dtflowcv.models.registry import file_sha256


def stage_model_artifact(
    *,
    source: str,
    name: str,
    version: str,
    out_dir: str | Path = "artifacts/models",
    expected_sha256: str | None = None,
    license_id: str = "",
    accept_license: bool = False,
) -> dict[str, Any]:
    if not accept_license:
        return blocked_payload(
            ["model_license_not_accepted: review the source/license and pass --accept-license"],
            source=source,
            license=license_id,
        )

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return blocked_payload(
            ["network_model_download_disabled: fetch externally, verify license/hash, then use model-register"],
            source=source,
            license=license_id,
        )

    source_path = Path(source)
    if not source_path.exists():
        return {"status": "failed", "errors": [f"missing_source_model:{source}"], "source": source}

    target_dir = Path(out_dir) / name / version
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source_path.name
    shutil.copy2(source_path, target)
    actual_sha = file_sha256(target)
    if expected_sha256 and actual_sha != expected_sha256:
        target.unlink(missing_ok=True)
        return {
            "status": "failed",
            "errors": [f"model_sha256_mismatch:expected:{expected_sha256}:actual:{actual_sha}"],
            "source": source,
        }
    return {
        "status": "ok",
        "source": source,
        "output": str(target),
        "sha256": actual_sha,
        "license": license_id,
        "claim_boundary": "Local artifact staging only; this does not validate model quality or deployment readiness.",
    }


__all__ = ["stage_model_artifact"]
