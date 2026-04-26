from __future__ import annotations

from typing import Any, Protocol


class Evaluator(Protocol):
    def reset(self) -> None: ...

    def process(self, sample: Any, prediction: Any) -> None: ...

    def evaluate(self) -> dict[str, Any]: ...


__all__ = ["Evaluator"]
