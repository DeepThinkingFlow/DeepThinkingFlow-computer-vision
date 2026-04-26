from __future__ import annotations

from typing import Any

from dtflowcv.dataset import ImageRecord, _duplicate_groups


def exact_duplicate_groups(records: list[ImageRecord], limit: int = 50) -> list[dict[str, Any]]:
    return _duplicate_groups(records, limit=limit)


__all__ = ["exact_duplicate_groups"]
