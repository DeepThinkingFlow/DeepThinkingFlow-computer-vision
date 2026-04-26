from __future__ import annotations

from pathlib import Path

from dtflowcv.models.registry import ModelRegistry, build_registry_entry, file_sha256


def test_model_registry_register_and_promote(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"not-a-real-model")
    registry_path = tmp_path / "registry.json"

    entry = build_registry_entry(
        name="road-smoke",
        version="0.1.0",
        path=artifact,
        license_id="Apache-2.0",
        source="unit-test",
        format="onnx",
    )
    result = ModelRegistry(registry_path).register(entry)

    assert result["status"] == "ok", result
    assert result["model"]["sha256"] == file_sha256(artifact)

    registry = ModelRegistry(registry_path)
    promote = registry.promote("road-smoke", "0.1.0", "validated")

    assert promote["status"] == "ok", promote
    assert ModelRegistry(registry_path).get("road-smoke", "0.1.0").status == "validated"  # type: ignore[union-attr]


def test_model_registry_rejects_missing_artifact(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    missing = tmp_path / "missing.onnx"

    entry = build_registry_entry(
        name="bad",
        version="0.1.0",
        path=missing,
        license_id="Apache-2.0",
        source="unit-test",
    )
    result = registry.register(entry)

    assert result["status"] == "failed"
    assert "missing_model_artifact" in result["errors"][0]
