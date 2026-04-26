from __future__ import annotations

import json
from typing import Any


def json_line(payload: dict[str, Any] | list[Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


__all__ = ["json_line"]
