from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dtflowcv.config import write_json
from dtflowcv.yolo import parse_yolo_label_file


@dataclass
class DatasetCard:
    """Machine-readable dataset card for reproducibility.

    Stores metadata about dataset version, content hash, class distribution,
    split statistics, and provenance.
    """
    name: str
    version: str = "1.0.0"
    created: str = ""
    source: str = ""
    description: str = ""
    sha256: str = ""
    image_count: int = 0
    label_count: int = 0
    annotation_count: int = 0
    invalid_label_count: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    split_stats: dict[str, int] = field(default_factory=dict)
    image_formats: list[str] = field(default_factory=list)
    total_size_bytes: int = 0
    annotation_format: str = "YOLO txt"
    license: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "created": self.created,
            "source": self.source,
            "description": self.description,
            "integrity": {
                "sha256": self.sha256,
                "total_size_bytes": self.total_size_bytes,
            },
            "content": {
                "image_count": self.image_count,
                "label_count": self.label_count,
                "annotation_count": self.annotation_count,
                "invalid_label_count": self.invalid_label_count,
                "annotation_format": self.annotation_format,
                "image_formats": self.image_formats,
            },
            "class_distribution": self.class_distribution,
            "split_stats": self.split_stats,
            "license": self.license,
            "notes": self.notes,
        }


def compute_dataset_hash(
    root: str | Path,
    include_images: bool = True,
    include_labels: bool = True,
) -> str:
    """Compute SHA-256 hash over all dataset files for reproducibility.

    Hashes filenames + contents of all images and labels, sorted for determinism.
    """
    root_path = Path(root).resolve()
    hasher = hashlib.sha256()

    patterns: list[str] = []
    if include_images:
        patterns.extend(["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"])
    if include_labels:
        patterns.extend(["*.txt"])

    all_files: list[Path] = []
    for pattern in patterns:
        all_files.extend(root_path.rglob(pattern))

    # Sort for deterministic hash
    all_files.sort(key=lambda p: str(p.relative_to(root_path)))

    for filepath in all_files:
        # Hash relative path
        rel = str(filepath.relative_to(root_path))
        hasher.update(rel.encode("utf-8"))
        # Hash file content
        hasher.update(filepath.read_bytes())

    return hasher.hexdigest()


def build_dataset_card(
    root: str | Path,
    name: str,
    class_names: list[str],
    source: str = "",
    description: str = "",
    version: str = "1.0.0",
    compute_hash: bool = True,
) -> DatasetCard:
    """Scan a YOLO dataset directory and build a DatasetCard.

    Expects structure:
        root/images/...
        root/labels/...
    """
    root_path = Path(root).resolve()
    images_dir = root_path / "images"
    labels_dir = root_path / "labels"

    # Count images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images: list[Path] = []
    if images_dir.exists():
        for ext in image_extensions:
            all_images.extend(images_dir.rglob(f"*{ext}"))

    # Count labels and annotations
    all_labels: list[Path] = []
    if labels_dir.exists():
        all_labels = list(labels_dir.rglob("*.txt"))

    annotation_count = 0
    invalid_label_count = 0
    class_dist: dict[str, int] = {name: 0 for name in class_names}
    for label_file in all_labels:
        try:
            boxes = parse_yolo_label_file(label_file)
        except ValueError:
            invalid_label_count += 1
            continue
        for box in boxes:
            annotation_count += 1
            if 0 <= box.class_id < len(class_names):
                class_dist[class_names[box.class_id]] += 1

    # Total size
    total_bytes = sum(f.stat().st_size for f in all_images) + sum(f.stat().st_size for f in all_labels)

    # Image formats
    formats = sorted(set(f.suffix.lower() for f in all_images))

    # Hash
    sha = ""
    if compute_hash:
        sha = compute_dataset_hash(root_path)

    return DatasetCard(
        name=name,
        version=version,
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source=source,
        description=description,
        sha256=sha,
        image_count=len(all_images),
        label_count=len(all_labels),
        annotation_count=annotation_count,
        invalid_label_count=invalid_label_count,
        class_distribution=class_dist,
        image_formats=formats,
        total_size_bytes=total_bytes,
    )


def write_dataset_card(
    card: DatasetCard,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write dataset card as both JSON and Markdown."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / "dataset_card.json"
    write_json(json_path, card.to_dict())

    # Markdown
    md_path = out / "DATASET_CARD.md"
    md = _card_to_markdown(card)
    md_path.write_text(md, encoding="utf-8")

    # SHA manifest
    sha_path = out / "manifest.sha256"
    sha_path.write_text(f"{card.sha256}  {card.name}\n", encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "sha256": str(sha_path),
    }


def verify_dataset_integrity(
    root: str | Path,
    expected_hash: str,
) -> dict[str, Any]:
    """Verify dataset integrity by re-computing hash."""
    actual = compute_dataset_hash(root)
    match = actual == expected_hash
    return {
        "verified": match,
        "expected_sha256": expected_hash,
        "actual_sha256": actual,
        "root": str(root),
    }


def _card_to_markdown(card: DatasetCard) -> str:
    lines = [
        f"# Dataset Card: {card.name}",
        "",
        f"**Version:** {card.version}",
        f"**Created:** {card.created}",
        f"**Source:** {card.source}" if card.source else "",
        f"**Description:** {card.description}" if card.description else "",
        "",
        "## Integrity",
        "",
        f"- SHA-256: `{card.sha256}`",
        f"- Total size: {card.total_size_bytes / (1024 * 1024):.1f} MB",
        "",
        "## Content",
        "",
        f"- Images: {card.image_count}",
        f"- Labels: {card.label_count}",
        f"- Annotations: {card.annotation_count}",
        f"- Invalid label files: {card.invalid_label_count}",
        f"- Format: {card.annotation_format}",
        f"- Image formats: {', '.join(card.image_formats)}",
        "",
        "## Class Distribution",
        "",
        "| Class | Count |",
        "|-------|-------|",
    ]
    for cls, count in card.class_distribution.items():
        lines.append(f"| {cls} | {count} |")

    if card.split_stats:
        lines.extend([
            "",
            "## Splits",
            "",
            "| Split | Images |",
            "|-------|--------|",
        ])
        for split, count in card.split_stats.items():
            lines.append(f"| {split} | {count} |")

    if card.license:
        lines.extend(["", f"**License:** {card.license}"])
    if card.notes:
        lines.extend(["", f"**Notes:** {card.notes}"])

    lines.append("")
    return "\n".join(lines)
