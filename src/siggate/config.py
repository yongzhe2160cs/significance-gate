"""``siggate.toml`` configuration: alpha, power, the metric, and gate strictness.

A repo drops a ``siggate.toml`` at its root (or points ``--config`` at one) to
fix the statistical thresholds and decide *which results should fail the build*.
Every field has a sensible default, so the file is optional.

Example ``siggate.toml``::

    [gate]
    alpha = 0.05
    power = 0.80
    mei = 0.05          # minimum effect of interest (metric units); NOISE vs UNDERPOWERED
    # What makes the CI check fail (exit nonzero). Any rule that matches blocks.
    fail_on = ["unsupported-improvement", "regression", "invalid"]

    [metric]
    name = "acc"        # lm-eval metric key / Inspect scorer name
    adapter = "auto"    # auto | lm-eval | inspect | raw

    [suite]
    correction = "holm" # holm (family-wise) | bh (false-discovery-rate)

    [selection]
    n_trials = 1        # best-of-N variants tried before reporting this one
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

__all__ = ["GateConfig", "FAIL_RULES", "load_config"]

# The strictness rules a `fail_on` list may reference.
FAIL_RULES = frozenset(
    {
        "unsupported-improvement",  # delta > 0 but the verdict is not REAL (a noisy win)
        "regression",  # a REAL effect in the wrong direction (candidate worse)
        "not-real",  # anything that is not a REAL effect
        "noise",  # verdict == NOISE
        "underpowered",  # verdict == UNDERPOWERED
        "invalid",  # verdict == INVALID (identical/degenerate runs)
    }
)

_DEFAULT_FAIL_ON = ("unsupported-improvement", "regression", "invalid")


@dataclass(frozen=True)
class GateConfig:
    """Resolved siggate configuration."""

    alpha: float = 0.05
    power: float = 0.80
    mei: float = 0.05  # minimum effect of interest (metric units); splits NOISE vs UNDERPOWERED
    fail_on: tuple[str, ...] = _DEFAULT_FAIL_ON
    metric: str = "acc"
    adapter: str = "auto"
    correction: str = "holm"
    n_trials: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"gate.alpha must be in (0, 1), got {self.alpha}")
        if not 0.0 < self.power < 1.0:
            raise ValueError(f"gate.power must be in (0, 1), got {self.power}")
        if self.mei <= 0.0:
            raise ValueError(f"gate.mei must be > 0, got {self.mei}")
        if self.correction not in ("holm", "bh"):
            raise ValueError(f"suite.correction must be 'holm' or 'bh', got {self.correction!r}")
        if self.n_trials < 1:
            raise ValueError(f"selection.n_trials must be >= 1, got {self.n_trials}")
        unknown = set(self.fail_on) - FAIL_RULES
        if unknown:
            raise ValueError(
                f"unknown gate.fail_on rule(s) {sorted(unknown)}; valid rules: {sorted(FAIL_RULES)}"
            )

    @classmethod
    def from_dict(cls, data: dict) -> GateConfig:
        """Build from a parsed TOML/dict, ignoring unknown top-level sections."""
        gate = data.get("gate", {}) or {}
        metric = data.get("metric", {}) or {}
        suite = data.get("suite", {}) or {}
        selection = data.get("selection", {}) or {}
        fail_on = gate.get("fail_on", _DEFAULT_FAIL_ON)
        if isinstance(fail_on, str):
            fail_on = [fail_on]
        return cls(
            alpha=float(gate.get("alpha", 0.05)),
            power=float(gate.get("power", 0.80)),
            mei=float(gate.get("mei", 0.05)),
            fail_on=tuple(fail_on),
            metric=str(metric.get("name", "acc")),
            adapter=str(metric.get("adapter", "auto")),
            correction=str(suite.get("correction", "holm")),
            n_trials=int(selection.get("n_trials", 1)),
        )

    def merged(self, **overrides: object) -> GateConfig:
        """Return a copy with the given (non-None) fields overridden — CLI flags win."""
        valid = {f.name for f in fields(self)}
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in valid:
                raise ValueError(f"unknown config override {key!r}")
            if key == "fail_on" and isinstance(value, Iterable) and not isinstance(value, str):
                value = tuple(value)
            data[key] = value
        return GateConfig(**data)


def load_config(path: str | Path | None = None) -> GateConfig:
    """Load a :class:`GateConfig` from ``path`` (or ``./siggate.toml`` if present).

    Returns all-defaults when no config file is found and ``path`` is None.
    Raises ``FileNotFoundError`` only when an explicit ``path`` does not exist.
    """
    if path is None:
        default = Path("siggate.toml")
        if not default.exists():
            return GateConfig()
        path = default
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    data = tomllib.loads(p.read_text())
    return GateConfig.from_dict(data)
