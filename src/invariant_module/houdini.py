from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Protocol


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@dataclass
class HoudiniCheckResult:
    valid: bool
    refuted: List[str] = field(default_factory=list)
    reason: str = ""


class HoudiniChecker(Protocol):
    def __call__(self, code: str, invariants: List[str]) -> HoudiniCheckResult:
        ...


class HoudiniFilter:
    """
    Verifier-backed Houdini filter.

    The checker is intentionally injected so experiments can use SeaHorn, Z3,
    Boogie, or a custom symbolic executor without changing the orchestration
    code. If no checker is provided, this class only normalizes the candidate
    list and acts as a pass-through placeholder.
    """

    def __init__(self, checker: HoudiniChecker | None = None, max_iterations: int = 20):
        self.checker = checker
        self.max_iterations = max_iterations
        self.last_trace: List[HoudiniCheckResult] = []

    def filter(self, code: str, candidates: List[str]) -> List[str]:
        current = dedupe_preserve_order(candidates)
        self.last_trace = []

        if self.checker is None:
            return current

        for _ in range(self.max_iterations):
            result = self.checker(code, current)
            self.last_trace.append(result)
            if result.valid:
                return current

            refuted = set(result.refuted)
            if not refuted:
                return current

            next_current = [inv for inv in current if inv not in refuted]
            if next_current == current:
                return current
            current = next_current

        return current
