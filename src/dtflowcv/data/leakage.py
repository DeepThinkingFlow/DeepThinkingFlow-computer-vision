from __future__ import annotations

from typing import Any

from dtflowcv.dataset import ImageRecord, _split_leakage_candidates


def split_leakage_candidates(records: list[ImageRecord], limit: int = 50) -> list[dict[str, Any]]:
    return _split_leakage_candidates(records, limit=limit)


__all__ = ["split_leakage_candidates"]
