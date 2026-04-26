from __future__ import annotations

import subprocess


def test_no_model_weights_are_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", "*.pt", "*.onnx", "*.engine", "*.torchscript"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == ""
