"""Gate decision: turn a verdict + ``fail_on`` rules into pass/fail and an exit code.

Separated from the statistics so the *policy* (what should break a build) is
explicit and independently testable. Exit codes:

* ``0`` — gate passed (nothing in ``fail_on`` matched).
* ``1`` — gate failed (a ``fail_on`` rule matched).
* ``2`` — usage / input error (raised by the CLI, not here).
"""

from __future__ import annotations

from collections.abc import Iterable

from .gate import INVALID, NOISE, REAL, UNDERPOWERED, SignificanceVerdict, SuiteVerdict

__all__ = ["failures", "exit_code", "suite_failures", "suite_exit_code"]


def failures(verdict: SignificanceVerdict, fail_on: Iterable[str]) -> list[str]:
    """Return the ``fail_on`` rules that this verdict trips (empty == gate passes)."""
    rules = set(fail_on)
    tripped: list[str] = []

    is_real = verdict.label == REAL
    if "unsupported-improvement" in rules and verdict.delta > 0 and not is_real:
        tripped.append("unsupported-improvement")
    if "regression" in rules and is_real and verdict.delta < 0:
        tripped.append("regression")
    if "not-real" in rules and not is_real:
        tripped.append("not-real")
    if "noise" in rules and verdict.label == NOISE:
        tripped.append("noise")
    if "underpowered" in rules and verdict.label == UNDERPOWERED:
        tripped.append("underpowered")
    if "invalid" in rules and verdict.label == INVALID:
        tripped.append("invalid")
    return tripped


def exit_code(verdict: SignificanceVerdict, fail_on: Iterable[str]) -> int:
    """``1`` if any ``fail_on`` rule trips for ``verdict``, else ``0``."""
    return 1 if failures(verdict, fail_on) else 0


def suite_failures(suite: SuiteVerdict, fail_on: Iterable[str]) -> dict[str, list[str]]:
    """Map each task that trips a rule to the rules it tripped."""
    out: dict[str, list[str]] = {}
    for task in suite.tasks:
        tripped = failures(task, fail_on)
        if tripped:
            out[task.name] = tripped
    return out


def suite_exit_code(suite: SuiteVerdict, fail_on: Iterable[str]) -> int:
    """``1`` if *any* task in the suite trips a ``fail_on`` rule, else ``0``."""
    return 1 if suite_failures(suite, fail_on) else 0
