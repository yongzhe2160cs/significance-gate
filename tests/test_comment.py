"""PR-comment rendering."""

from __future__ import annotations

import siggate
from siggate.comment import COMMENT_MARKER, render_comment, render_markdown, render_suite_comment


def test_comment_has_marker_and_label(real_case):
    v = siggate.compare(*real_case, name="mmlu")
    md = render_comment(v)
    assert md.startswith(COMMENT_MARKER)
    assert "Significance Gate — REAL" in md
    assert "🟢" in md
    assert "`mmlu`" in md
    assert "| p-value |" in md  # the numbers table


def test_underpowered_comment_shows_more_samples(underpowered_case):
    v = siggate.compare(*underpowered_case, name="t")
    md = render_comment(v)
    assert "Underpowered" in md
    assert "more samples" in md
    assert str(v.n_more) in md.replace(",", "")  # the actual number appears


def test_noise_comment_says_within_noise(noise_case):
    v = siggate.compare(*noise_case, name="t")
    md = render_comment(v)
    assert "Within noise" in md or "within noise" in md
    assert "⚪" in md


def test_suite_comment_lists_survivors():
    from tests.conftest import _paired

    suite = {
        "mmlu": _paired(1200, 0.70, 0.78, seed=10),
        "gsm8k": _paired(1200, 0.64, 0.643, seed=11),
    }
    sv = siggate.compare_suite(suite)
    md = render_suite_comment(sv)
    assert COMMENT_MARKER in md
    assert "multiplicity correction" in md or "correction" in md
    assert "`mmlu`" in md
    assert "| task | delta |" in md


def test_render_markdown_dispatch(real_case):
    from tests.conftest import _paired

    v = siggate.compare(*real_case, name="t")
    assert "Significance Gate — REAL" in render_markdown(v)
    sv = siggate.compare_suite({"a": _paired(800, 0.7, 0.78, seed=1)})
    assert "suite" in render_markdown(sv)
