"""siggate — the vendor-neutral significance gate for AI evals.

Most eval tooling shows a number moving between two runs. None of it tells you
whether the move is *real*. siggate does: on every eval run or model comparison
it reports whether a reported improvement is statistically REAL, within NOISE,
or simply UNDERPOWERED at this sample size — and, when underpowered, how many
more samples you'd need to call it. It runs as a CLI, a CI exit-code gate, and a
GitHub Action that posts the verdict as a PR comment.

The statistics are not reimplemented here. siggate is the productization layer
over the author's :mod:`deltagate` library (the eval-reliability toolkit): the
paired-delta confidence interval, multiple-comparison correction, power /
minimum-sample math, and the selection-bias-aware "is this real?" probability
all come from there.

Quickstart::

    import siggate
    v = siggate.compare(baseline_scores, candidate_scores, name="mmlu")
    print(v.summary())          # one-line human verdict
    print(v.label)              # REAL / NOISE / UNDERPOWERED / INVALID

See also :func:`siggate.compare_suite` for a multi-task comparison with
Holm/Benjamini-Hochberg multiplicity correction.
"""

from __future__ import annotations

from .comment import render_comment, render_markdown, render_suite_comment
from .config import GateConfig, load_config
from .decision import exit_code, failures, suite_exit_code, suite_failures
from .gate import (
    INVALID,
    LABELS,
    NOISE,
    REAL,
    UNDERPOWERED,
    SignificanceVerdict,
    SuiteVerdict,
    compare,
    compare_suite,
    verdict_from_report,
)

__version__ = "0.1.1"

__all__ = [
    "__version__",
    # core gate
    "compare",
    "compare_suite",
    "verdict_from_report",
    "SignificanceVerdict",
    "SuiteVerdict",
    # labels
    "REAL",
    "NOISE",
    "UNDERPOWERED",
    "INVALID",
    "LABELS",
    # decision / exit codes
    "failures",
    "exit_code",
    "suite_failures",
    "suite_exit_code",
    # config
    "GateConfig",
    "load_config",
    # comment rendering
    "render_comment",
    "render_suite_comment",
    "render_markdown",
]
