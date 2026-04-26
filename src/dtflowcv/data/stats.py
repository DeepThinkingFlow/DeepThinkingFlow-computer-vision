from __future__ import annotations

from collections import Counter
from typing import Any

from dtflowcv.dataset import _class_imbalance_severity


def class_imbalance_severity(class_counter: Counter[int]) -> dict[str, Any]:
    return _class_imbalance_severity(class_counter)


__all__ = ["class_imbalance_severity"]
