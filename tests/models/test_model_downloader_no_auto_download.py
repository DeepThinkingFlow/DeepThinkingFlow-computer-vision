from __future__ import annotations

from pathlib import Path

from dtflowcv.models.downloader import stage_model_artifact


def test_model_download_requires_explicit_license_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"model")

    result = stage_model_artifact(
        source=str(source),
        name="road-smoke",
        version="0.1.0",
        out_dir=tmp_path / "out",
        license_id="Apache-2.0",
        accept_license=False,
    )

    assert result["status"] == "blocked"
    assert "model_license_not_accepted" in result["build_blockers"][0]


def test_model_download_never_fetches_network_url(tmp_path: Path) -> None:
    result = stage_model_artifact(
        source="https://example.invalid/model.onnx",
        name="road-smoke",
        version="0.1.0",
        out_dir=tmp_path / "out",
        license_id="Apache-2.0",
        accept_license=True,
    )

    assert result["status"] == "blocked"
    assert "network_model_download_disabled" in result["build_blockers"][0]


def test_model_download_stages_local_artifact_after_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"model")

    result = stage_model_artifact(
        source=str(source),
        name="road-smoke",
        version="0.1.0",
        out_dir=tmp_path / "out",
        license_id="Apache-2.0",
        accept_license=True,
    )

    assert result["status"] == "ok", result
    assert Path(result["output"]).exists()
