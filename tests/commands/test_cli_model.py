from __future__ import annotations

import json
from pathlib import Path

from dtflowcv.commands.app import app
from typer.testing import CliRunner


def test_model_info_empty_registry_is_stable_json(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["model-info", "--registry", str(tmp_path / "registry.json")])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["models"] == []


def test_model_download_without_license_acceptance_is_blocked(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"model")

    result = CliRunner().invoke(
        app,
        [
            "model-download",
            "--source",
            str(source),
            "--name",
            "road-smoke",
            "--version",
            "0.1.0",
            "--out",
            str(tmp_path / "out"),
            "--license",
            "Apache-2.0",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["status"] == "blocked"
