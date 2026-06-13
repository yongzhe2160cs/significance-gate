"""Gate decision / exit-code policy."""

from __future__ import annotations

from dataclasses import replace

import pytest

import siggate
from siggate.decision import exit_code, failures, suite_exit_code
from siggate.gate import NOISE, REAL, UNDERPOWERED


def _verdict(label, delta):
    """A minimal verdict with just the fields the decision logic reads."""
    base = siggate.compare([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0], name="x")
    return replace(base, label=label, delta=delta)


def test_unsupported_improvement_fails():
    v = _verdict(NOISE, delta=0.02)  # claimed +2% but only noise
    assert failures(v, ["unsupported-improvement"]) == ["unsupported-improvement"]
    assert exit_code(v, ["unsupported-improvement"]) == 1


def test_real_improvement_passes():
    v = _verdict(REAL, delta=0.02)
    assert failures(v, ["unsupported-improvement"]) == []
    assert exit_code(v, ["unsupported-improvement"]) == 0


def test_negative_noise_is_not_unsupported_improvement():
    # No improvement claimed (delta < 0), so this rule does not fire.
    v = _verdict(NOISE, delta=-0.02)
    assert failures(v, ["unsupported-improvement"]) == []


def test_regression_rule_fires_on_real_negative():
    v = _verdict(REAL, delta=-0.03)
    assert "regression" in failures(v, ["regression"])
    assert exit_code(v, ["regression"]) == 1


def test_not_real_rule_is_strict():
    assert exit_code(_verdict(UNDERPOWERED, 0.0), ["not-real"]) == 1
    assert exit_code(_verdict(REAL, 0.05), ["not-real"]) == 0


def test_empty_fail_on_always_passes():
    assert exit_code(_verdict(NOISE, 0.02), []) == 0
    assert exit_code(_verdict(UNDERPOWERED, 0.0), []) == 0


def test_default_rules_block_noisy_win_but_pass_real(real_case, noise_case):
    rules = ["unsupported-improvement", "regression", "invalid"]
    real = siggate.compare(*real_case, name="t")
    noise = siggate.compare(*noise_case, name="t")
    assert exit_code(real, rules) == 0
    # noise_case delta may be ~0; force a claimed improvement to exercise the rule
    assert exit_code(replace(noise, delta=0.02), rules) == 1


def test_suite_exit_code_fails_if_any_task_fails():
    from tests.conftest import _paired

    suite = {
        "win": _paired(1200, 0.70, 0.78, seed=1),
        "noisy_claim": _paired(1500, 0.72, 0.724, seed=2),
    }
    sv = siggate.compare_suite(suite)
    # Force the noisy task to look like a claimed improvement.
    sv.tasks[:] = [replace(t, delta=0.02) if t.name == "noisy_claim" else t for t in sv.tasks]
    assert suite_exit_code(sv, ["unsupported-improvement"]) == 1


@pytest.mark.parametrize("rule", ["noise", "underpowered", "invalid"])
def test_label_rules(rule):
    label = {"noise": NOISE, "underpowered": UNDERPOWERED, "invalid": "INVALID"}[rule]
    assert exit_code(_verdict(label, 0.0), [rule]) == 1
