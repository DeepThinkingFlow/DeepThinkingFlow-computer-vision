from pathlib import Path

from dtflowcv.dataset import audit_dataset, load_records, stratified_split_records, write_split_manifests
from dtflowcv.demo import create_demo_dataset


def test_demo_dataset_audit_and_split(tmp_path: Path) -> None:
    classes = ["person", "car", "bus"]
    dataset = tmp_path / "demo"
    create_demo_dataset(dataset, classes, image_count=15, seed=7)

    report = audit_dataset(dataset, classes)
    assert report["summary"]["image_count"] == 15
    assert report["summary"]["object_count"] > 15
    assert report["summary"]["missing_label_files"] == 0

    records = load_records(dataset)
    splits = stratified_split_records(records, {"train": 0.60, "val": 0.20, "test": 0.20}, seed=7)
    assert sum(len(value) for value in splits.values()) == 15
    assert {key: len(value) for key, value in splits.items()} == {"train": 9, "val": 3, "test": 3}

    out = tmp_path / "splits"
    write_split_manifests(splits, out)
    assert (out / "train.txt").exists()
    assert (out / "split_summary.json").exists()
