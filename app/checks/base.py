"""Checker plugin interface — v1 ships no implementations.

A checker turns the list of detected :class:`Item` objects into a list of
human-readable :class:`CheckResult` findings. This is the seam where the
"AINative 审查层" (probability sum, six-finger, compliance, etc.) plugs in.

Design notes:
- Checkers receive the full item list (text + codes + polygons), never the raw
  image, so they're cheap to run in batch and easy to unit-test.
- A checker that genuinely needs the image (e.g. six-finger detection) gets the
  image path passed at registration — see :func:`Checker.__init__` contract
  in v2.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal

from ..schemas import Item

Severity = Literal["info", "warning", "error"]


@dataclass
class CheckResult:
    severity: Severity
    message: str
    item_ids: list[str] = field(default_factory=list)


class Checker(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def run(self, items: list[Item]) -> list[CheckResult]:
        raise NotImplementedError


def run_all(checkers: list[Checker], items: list[Item]) -> list[CheckResult]:
    """Run every checker, concatenating results. A failing checker is skipped."""
    out: list[CheckResult] = []
    for c in checkers:
        try:
            out.extend(c.run(items))
        except Exception:  # noqa: BLE001 — one bad checker must not kill the run
            pass
    return out
