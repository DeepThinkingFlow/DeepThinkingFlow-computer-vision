from __future__ import annotations

import json
from pathlib import Path

from dtflowcv.core.vendor import vendor_check


def test_forbidden_license_in_permissive_zone_fails(tmp_path: Path) -> None:
    component = tmp_path / "third_party" / "permissive" / "bad_component"
    component.mkdir(parents=True)
    (component / "LICENSE").write_text("AGPL-3.0\n", encoding="utf-8")
    (component / "UPSTREAM.md").write_text("# Upstream\n", encoding="utf-8")
    (component / "PATCHES.md").write_text("# Local Patches\n", encoding="utf-8")
    manifest = {
        "manifest_version": 1,
        "components": [
            {
                "name": "bad_component",
                "upstream_url": "https://example.invalid/bad",
                "upstream_commit": "abc123",
                "license": "AGPL-3.0",
                "vendored_path": "third_party/permissive/bad_component",
                "reason": "negative test",
                "local_modifications": False,
                "patches_file": "third_party/permissive/bad_component/PATCHES.md",
                "owner": "DeepThinkingFlow",
                "allowed_in_core": False,
            }
        ],
    }
    (tmp_path / "third_party" / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = vendor_check(tmp_path)

    assert result["status"] == "failed"
    assert result["forbidden_components"] == 1
    assert any("forbidden_license:AGPL-3.0" in error for error in result["errors"])
