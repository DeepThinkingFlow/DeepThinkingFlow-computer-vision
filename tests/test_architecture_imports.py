from __future__ import annotations

from pathlib import Path


def test_cli_shim_uses_commands_app() -> None:
    from dtflowcv.cli import app as cli_app
    from dtflowcv.commands.app import app as commands_app
    from dtflowcv.commands.data import make_demo_dataset

    assert cli_app is commands_app
    assert make_demo_dataset


def test_new_architecture_boundaries_reexport_existing_contracts(tmp_path: Path) -> None:
    from dtflowcv.core.hashing import sha256_text
    from dtflowcv.data.yolo import parse_yolo_label_file
    from dtflowcv.evaluation.coco_ap import coco_style_metrics
    from dtflowcv.evaluation.iou import box_iou
    from dtflowcv.models.registry import ModelRegistry
    from dtflowcv.runtime.tracking import SORTTracker

    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    assert sha256_text("dtflowcv")
    assert parse_yolo_label_file(label)[0].class_id == 0
    assert box_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert callable(coco_style_metrics)
    assert ModelRegistry
    assert SORTTracker
