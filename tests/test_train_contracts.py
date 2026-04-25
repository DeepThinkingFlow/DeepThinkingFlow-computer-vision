from pathlib import Path

import yaml
from dtflowcv.train import _dataset_blockers, _runtime_blockers


def test_runtime_blockers_allow_cpu_device() -> None:
    assert _runtime_blockers({"training": {"device": "cpu"}}) == []


def test_runtime_blockers_fail_closed_for_requested_cuda_without_cuda() -> None:
    blockers = _runtime_blockers({"training": {"device": 0}})
    assert not blockers or blockers[0].startswith("gpu_device_requested_but_")


def test_dataset_blockers_accept_ultralytics_names_list(tmp_path: Path) -> None:
    image = tmp_path / "one.jpg"
    image.write_bytes(b"placeholder")
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text(str(image) + "\n", encoding="utf-8")
    val.write_text(str(image) + "\n", encoding="utf-8")
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump({"path": ".", "train": "train.txt", "val": "val.txt", "names": ["person", "car"]}),
        encoding="utf-8",
    )

    assert _dataset_blockers(dataset_yaml, expected_class_count=2) == []
