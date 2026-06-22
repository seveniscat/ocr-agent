"""Checker sub-package (v1: interface only).

Each :class:`Checker` inspects the aggregated :class:`Item` list and emits
findings (e.g. "probability sum != 1", "AI-generated hand has 6 fingers").
v1 ships none; the pipeline leaves the hook in place so v2 can plug them in
without touching orchestration.
"""
from .base import Checker, CheckResult, run_all  # noqa: F401
