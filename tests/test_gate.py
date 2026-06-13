"""Gate logic: labelling, the samples-needed line, summaries, and the suite path.

The deep statistics are already tested in deltagate; here we test the
*productization* — that the verdict labels and human-facing strings behave.
"""

from __future__ import annotations

import siggate
from siggate.gate import INVALID, NOISE, REAL, UNDERPOWERED


def test_real_case_is_real(real_case):
    base, cand = real_case
    v = siggate.compare(base, cand, name="t")
    assert v.label == REAL
    assert v.passed
    assert v.delta > 0
    assert v.p_value < 0.05
    assert v.emoji == "🟢"


def test_noise_case_is_noise(noise_case):
    base, cand = noise_case
    v = siggate.compare(base, cand, name="t")
    assert v.label == NOISE
    assert not v.passed
    # Well-powered: the eval could resolve a mei-sized effect.
    assert v.min_detectable <= v.mei
    assert "within noise" in v.summary()


def test_underpowered_case_needs_more_samples(underpowered_case):
    base, cand = underpowered_case
    v = siggate.compare(base, cand, name="t")
    assert v.label == UNDERPOWERED
    # Could NOT resolve a mei-sized effect at this n.
    assert v.min_detectable > v.mei
    assert v.n_needed is not None and v.n_needed > v.n
    assert v.n_more == v.n_needed - v.n
    assert "more samples" in v.summary()


def test_mei_controls_noise_vs_underpowered(noise_case):
    base, cand = noise_case
    # With a tiny mei, the well-powered null is no longer "powered enough".
    strict = siggate.compare(base, cand, name="t", mei=0.001)
    assert strict.label == UNDERPOWERED
    lax = siggate.compare(base, cand, name="t", mei=0.05)
    assert lax.label == NOISE


def test_identical_runs_are_invalid():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    v = siggate.compare(scores, list(scores), name="dup")
    assert v.label == INVALID
    assert v.emoji == "🔴"
    assert "identical" in v.summary().lower()
    assert v.flags  # carries the deltagate red flag


def test_direction_is_candidate_minus_baseline():
    # candidate strictly worse than baseline -> negative delta.
    baseline = [1.0] * 50 + [0.0] * 50
    candidate = [1.0] * 40 + [0.0] * 60
    v = siggate.compare(baseline, candidate, name="t")
    assert v.delta < 0
    assert v.baseline_mean > v.candidate_mean


def test_to_dict_is_json_safe(real_case):
    import json

    base, cand = real_case
    v = siggate.compare(base, cand, name="t")
    d = v.to_dict()
    assert "report" not in d
    assert d["label"] == REAL
    assert d["passed"] is True
    json.dumps(d)  # must not raise


def test_suite_multiplicity_culls_lucky_wins():
    # mmlu is a real win; the other three are null. With many tasks, a naive
    # scan finds spurious wins that Holm correction removes.
    from tests.conftest import _paired

    suite = {
        "mmlu": _paired(1200, 0.70, 0.77, seed=10),
        "gsm8k": _paired(1200, 0.64, 0.643, seed=11),
        "arc": _paired(1200, 0.81, 0.815, seed=12),
        "hellaswag": _paired(1200, 0.88, 0.885, seed=13),
    }
    sv = siggate.compare_suite(suite, method="holm")
    assert [t.name for t in sv.survivors] == ["mmlu"]
    # Survivors are a subset of naive wins (correction only removes).
    assert set(t.name for t in sv.survivors).issubset(set(sv.naive_wins))
    assert all(t.corrected for t in sv.tasks)


def test_suite_bh_at_least_as_permissive_as_holm():
    from tests.conftest import _paired

    suite = {f"task{i}": _paired(800, 0.70, 0.74, seed=20 + i) for i in range(6)}
    holm = siggate.compare_suite(suite, method="holm")
    bh = siggate.compare_suite(suite, method="bh")
    assert len(bh.survivors) >= len(holm.survivors)
