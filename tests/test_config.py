"""Config loading, validation, and CLI-override merging."""

from __future__ import annotations

import pytest

from siggate.config import GateConfig, load_config


def test_defaults():
    cfg = GateConfig()
    assert cfg.alpha == 0.05
    assert cfg.power == 0.80
    assert cfg.mei == 0.05
    assert cfg.correction == "holm"
    assert "unsupported-improvement" in cfg.fail_on


def test_load_from_toml(tmp_path):
    p = tmp_path / "siggate.toml"
    p.write_text(
        "[gate]\nalpha = 0.01\npower = 0.9\nmei = 0.02\n"
        'fail_on = ["not-real"]\n\n'
        '[metric]\nname = "exact_match"\nadapter = "lm-eval"\n\n'
        '[suite]\ncorrection = "bh"\n\n'
        "[selection]\nn_trials = 8\n"
    )
    cfg = load_config(p)
    assert cfg.alpha == 0.01
    assert cfg.power == 0.9
    assert cfg.mei == 0.02
    assert cfg.fail_on == ("not-real",)
    assert cfg.metric == "exact_match"
    assert cfg.adapter == "lm-eval"
    assert cfg.correction == "bh"
    assert cfg.n_trials == 8


def test_missing_default_config_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config(None) == GateConfig()


def test_explicit_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_string_fail_on_is_wrapped(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[gate]\nfail_on = "regression"\n')
    assert load_config(p).fail_on == ("regression",)


def test_unknown_fail_on_rule_rejected():
    with pytest.raises(ValueError, match="unknown gate.fail_on"):
        GateConfig(fail_on=("not-a-rule",))


@pytest.mark.parametrize(
    "bad", [{"alpha": 0.0}, {"power": 1.0}, {"mei": 0.0}, {"correction": "x"}, {"n_trials": 0}]
)
def test_validation(bad):
    with pytest.raises(ValueError):
        GateConfig(**bad)


def test_merged_overrides_only_non_none():
    cfg = GateConfig()
    merged = cfg.merged(alpha=0.01, power=None, fail_on=[])
    assert merged.alpha == 0.01
    assert merged.power == cfg.power  # None override ignored
    assert merged.fail_on == ()
