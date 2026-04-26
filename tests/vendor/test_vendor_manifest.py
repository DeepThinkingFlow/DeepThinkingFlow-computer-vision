from __future__ import annotations

from pathlib import Path

from dtflowcv.core.vendor import license_check, vendor_check


def test_empty_vendor_manifest_is_valid_for_repo_root() -> None:
    result = vendor_check(Path.cwd())

    assert result["status"] == "ok", result
    assert result["vendored_components"] == 0
    assert result["forbidden_components"] == 0


def test_license_check_accepts_apache_core_and_empty_vendor_manifest() -> None:
    result = license_check(Path.cwd())

    assert result["status"] == "ok", result
    assert result["project_license"] == "Apache-2.0"
    assert result["vendored_components"] == 0
