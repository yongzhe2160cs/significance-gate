"""The gate: turn two aligned eval runs into a clear REAL / NOISE / UNDERPOWERED verdict.

This is the productization layer. The statistics — paired-delta CIs, the
selection-bias-aware "is this real?" probability, multiplicity correction, and
the power / minimum-sample math — all live in the author's :mod:`deltagate`
library (the eval-reliability toolkit) and are *reused here verbatim*. siggate's
job is to wrap that decomposable statistical report in a single labelled verdict
with a one-line human summary and the killer "you'd need N more samples" line,
so it can drive a CI check and a PR comment.

Convention: ``compare(baseline, candidate)`` judges the *candidate* against the
*baseline*. The reported delta is ``mean(candidate) - mean(baseline)`` — positive
means the candidate improved.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

from deltagate import evaluate_comparison, min_samples_for_delta, reliability_report
from deltagate.report import ComparisonReport, SuiteReport

__all__ = [
    "SignificanceVerdict",
    "SuiteVerdict",
    "REAL",
    "NOISE",
    "UNDERPOWERED",
    "INVALID",
    "LABELS",
    "compare",
    "compare_suite",
    "verdict_from_report",
]

REAL = "REAL"
NOISE = "NOISE"
UNDERPOWERED = "UNDERPOWERED"
INVALID = "INVALID"
LABELS = (REAL, NOISE, UNDERPOWERED, INVALID)

_EMOJI = {REAL: "🟢", UNDERPOWERED: "🟡", NOISE: "⚪", INVALID: "🔴"}

ScoresLike = Sequence[float] | Mapping[object, float]


def _fmt(value: float, proportion: bool, *, signed: bool = True) -> str:
    """Format a delta/score for humans: percentage points if it's a proportion."""
    if proportion:
        s = f"{value * 100:+.2f}pp" if signed else f"{value * 100:.2f}%"
    else:
        s = f"{value:+.4f}" if signed else f"{value:.4f}"
    return s


@dataclass(frozen=True)
class SignificanceVerdict:
    """One labelled verdict for a single A-vs-B paired comparison.

    Wraps a :class:`deltagate.report.ComparisonReport` (kept in ``report``) with
    the gate label and the productized "samples needed" guidance.
    """

    name: str
    label: str  # one of LABELS
    delta: float
    baseline_mean: float
    candidate_mean: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n: int
    alpha: float
    power: float
    mei: float
    proportion: bool
    min_detectable: float
    n_needed: int | None  # paired samples to detect the *observed* delta at `power`
    n_more: int | None  # additional samples beyond the current n
    prob_real: float
    deflated: float | None
    n_trials: int
    flags: list[tuple[str, str]] = field(default_factory=list)
    # Suite context (set by compare_suite when multiplicity correction applies):
    corrected: bool = False
    adjusted_p: float | None = None
    correction_method: str | None = None
    report: ComparisonReport | None = field(default=None, repr=False, compare=False)

    @property
    def emoji(self) -> str:
        return _EMOJI[self.label]

    @property
    def passed(self) -> bool:
        """True iff the verdict is a statistically real effect."""
        return self.label == REAL

    @property
    def is_improvement(self) -> bool:
        return self.delta > 0

    def summary(self) -> str:
        """The one-line pitch: what moved, and whether it's real."""
        d = _fmt(self.delta, self.proportion)
        p = f"p={self.p_value:.3g}"
        if self.label == INVALID:
            msg = self.flags[0][1] if self.flags else "the two runs are identical"
            return f"{self.emoji} INVALID: {msg}."
        if self.label == REAL:
            lo = _fmt(self.ci_lower, self.proportion)
            hi = _fmt(self.ci_upper, self.proportion)
            ci = f"95% CI [{lo}, {hi}]"
            extra = ""
            if self.corrected:
                extra = (
                    f", survives {self.correction_method} correction (adj p={self.adjusted_p:.3g})"
                )
            return (
                f"{self.emoji} REAL: candidate {d} vs baseline ({p}, {ci}){extra} "
                f"— a real change at α={self.alpha:g}."
            )
        if self.label == UNDERPOWERED:
            need = self._needs_clause()
            return f"{self.emoji} UNDERPOWERED: {d} is within noise ({p}); {need}"
        # NOISE
        powered = _fmt(self.mei, self.proportion, signed=False)
        return (
            f"{self.emoji} NOISE: {d} is within noise ({p}); n={self.n:,} was enough to "
            f"detect a {powered} effect and none is there."
        )

    def _needs_clause(self) -> str:
        if self.n_more is None or self.n_needed is None:
            return "more samples needed to resolve it."
        if self.n_more <= 0:
            return (
                f"min detectable delta at n={self.n} is "
                f"{_fmt(self.min_detectable, self.proportion, signed=False)}."
            )
        return (
            f"you'd need ~{self.n_more:,} more samples "
            f"({self.n_needed:,} total) to call it at {self.power:.0%} power."
        )

    def render(self) -> str:
        """Compact multi-line text report (CLI ``--format text``)."""
        lines = [
            f"{self.emoji} {self.name}: {self.label}",
            f"   baseline {_fmt(self.baseline_mean, self.proportion, signed=False)}  "
            f"candidate {_fmt(self.candidate_mean, self.proportion, signed=False)}  "
            f"delta {_fmt(self.delta, self.proportion)}  (n={self.n})",
            f"   paired CI [{_fmt(self.ci_lower, self.proportion)}, "
            f"{_fmt(self.ci_upper, self.proportion)}]  p={self.p_value:.4g}  "
            f"P(real)={self.prob_real:.2f}"
            + (
                f"  deflated(best-of-{self.n_trials})={self.deflated:.2f}"
                if self.deflated is not None
                else ""
            ),
            f"   min detectable @ n={self.n}: "
            f"{_fmt(self.min_detectable, self.proportion, signed=False)}"
            + (
                f"   -> ~{self.n_more:,} more samples for an effect this size"
                if self.n_more and self.label == UNDERPOWERED
                else ""
            ),
        ]
        if self.corrected:
            lines.append(f"   {self.correction_method} adjusted p={self.adjusted_p:.4g}")
        for name, message in self.flags:
            lines.append(f"   RED FLAG [{name}]: {message}")
        lines.append(f"   => {self.summary()}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serializable dict (drops the heavy embedded report)."""
        d = asdict(self)
        d.pop("report", None)
        d["emoji"] = self.emoji
        d["passed"] = self.passed
        d["summary"] = self.summary()
        return d


def _samples_needed(
    delta: float, stderr: float, n: int, power: float, alpha: float
) -> tuple[int | None, int | None]:
    """How many paired samples to detect the *observed* delta at `power`.

    Recovers ``sd_diff`` from the paired standard error (``se = sd_diff/sqrt(n)``)
    — the same standard deviation deltagate's power math consumes — then asks
    deltagate for the minimum sample size.
    """
    if delta == 0.0 or stderr <= 0.0 or n < 2:
        return None, None
    sd_diff = stderr * math.sqrt(n)
    try:
        needed = min_samples_for_delta(delta, sd_diff, power=power, alpha=alpha)
    except ValueError:
        return None, None
    return needed, max(0, needed - n)


def verdict_from_report(
    report: ComparisonReport,
    power: float = 0.8,
    *,
    mei: float = 0.05,
    corrected_significant: bool | None = None,
    adjusted_p: float | None = None,
    correction_method: str | None = None,
) -> SignificanceVerdict:
    """Label a deltagate :class:`ComparisonReport` as REAL / NOISE / UNDERPOWERED / INVALID.

    The NOISE-vs-UNDERPOWERED split needs an equivalence margin — the *minimum
    effect of interest* ``mei`` (in the metric's own units; default 0.05, i.e.
    5 points for an accuracy-style proportion). A non-significant result is
    **NOISE** when the eval was powered to find a ``mei``-sized effect
    (``min_detectable <= mei``) and found none, and **UNDERPOWERED** when it
    could not even resolve a ``mei``-sized effect (``min_detectable > mei``) —
    the case where collecting more samples is the right next step.

    Pass ``corrected_significant`` (and the adjusted p / method) when a suite-level
    multiplicity correction has been applied; the label then reflects survival of
    that correction rather than the naive per-task test.
    """
    identical = any(f.name == "identical_runs" for f in report.flags)
    significant = report.significant if corrected_significant is None else corrected_significant

    if identical:
        label = INVALID
    elif significant:
        label = REAL
    elif report.min_detectable > mei:
        # Could not even resolve a meaningful (mei-sized) effect at this n.
        label = UNDERPOWERED
    else:
        # Powered to find a meaningful effect; none is distinguishable from zero.
        label = NOISE

    proportion = 0.0 <= report.mean_a <= 1.0 and 0.0 <= report.mean_b <= 1.0
    n_needed, n_more = _samples_needed(
        report.delta, report.paired["stderr"], report.n, power, report.alpha
    )
    return SignificanceVerdict(
        name=report.name,
        label=label,
        delta=report.delta,
        baseline_mean=report.mean_b,
        candidate_mean=report.mean_a,
        ci_lower=report.paired["lower"],
        ci_upper=report.paired["upper"],
        p_value=report.p_value,
        n=report.n,
        alpha=report.alpha,
        power=power,
        mei=mei,
        proportion=proportion,
        min_detectable=report.min_detectable,
        n_needed=n_needed,
        n_more=n_more,
        prob_real=report.prob_real,
        deflated=report.deflated,
        n_trials=report.n_trials,
        flags=[(f.name, f.message) for f in report.flags],
        corrected=corrected_significant is not None,
        adjusted_p=adjusted_p,
        correction_method=correction_method,
        report=report,
    )


def compare(
    baseline: ScoresLike,
    candidate: ScoresLike,
    name: str = "comparison",
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    mei: float = 0.05,
    n_trials: int = 1,
    trial_deltas: Sequence[float] | None = None,
    n_boot: int = 5000,
    seed: int = 0,
) -> SignificanceVerdict:
    """Gate a single candidate-vs-baseline comparison on shared samples.

    ``baseline`` and ``candidate`` are per-sample scores — sequences already
    aligned by sample, or ``{sample_id: score}`` mappings (aligned by id). The
    reported delta is candidate minus baseline.
    """
    # deltagate's evaluate_comparison reports delta = mean(a) - mean(b); pass
    # candidate as A and baseline as B so delta reads as the improvement.
    report = evaluate_comparison(
        candidate,
        baseline,
        name=name,
        level=1.0 - alpha,
        alpha=alpha,
        n_trials=n_trials,
        trial_deltas=trial_deltas,
        n_boot=n_boot,
        seed=seed,
    )
    return verdict_from_report(report, power=power, mei=mei)


@dataclass(frozen=True)
class SuiteVerdict:
    """Suite-level verdict: per-task verdicts plus multiplicity-corrected survivors."""

    tasks: list[SignificanceVerdict]
    method: str  # "holm" or "bh"
    alpha: float
    power: float
    suite: SuiteReport | None = field(default=None, repr=False, compare=False)

    @property
    def survivors(self) -> list[SignificanceVerdict]:
        """Tasks that are REAL after multiplicity correction."""
        return [t for t in self.tasks if t.label == REAL]

    @property
    def naive_wins(self) -> list[str]:
        return list(self.suite.naive_wins) if self.suite is not None else []

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "power": self.power,
            "n_tasks": len(self.tasks),
            "survivors": [t.name for t in self.survivors],
            "naive_wins": self.naive_wins,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    def render(self) -> str:
        lines = [t.render() for t in self.tasks]
        lines.append("")
        lines.append(
            f"suite: {len(self.tasks)} tasks  alpha={self.alpha:g}  correction={self.method}"
        )
        lines.append(f"   naive per-task wins : {len(self.naive_wins)} {self.naive_wins}")
        surv = [t.name for t in self.survivors]
        lines.append(f"   survive correction : {len(surv)} {surv}")
        return "\n".join(lines)


def compare_suite(
    suite: Mapping[str, tuple[ScoresLike, ScoresLike]],
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    mei: float = 0.05,
    method: str = "holm",
    n_boot: int = 5000,
    seed: int = 0,
) -> SuiteVerdict:
    """Gate a whole suite of tasks with multiplicity correction.

    ``suite`` maps ``task_name -> (baseline_scores, candidate_scores)``. Holm
    (family-wise) or Benjamini-Hochberg (FDR) correction is applied across the
    per-task paired p-values; a task is labelled REAL only if it survives both
    the naive test *and* the chosen correction.
    """
    if method not in ("holm", "bh"):
        raise ValueError(f"method must be 'holm' or 'bh', got {method!r}")

    # Pass candidate as A, baseline as B (delta = candidate - baseline), matching
    # `compare`. deltagate preserves input order in the corrections.
    dg_suite = {task: (candidate, baseline) for task, (baseline, candidate) in suite.items()}
    sr = reliability_report(dg_suite, level=1.0 - alpha, alpha=alpha, n_boot=n_boot, seed=seed)

    correction = sr.holm if method == "holm" else sr.bh
    method_name = correction.method
    verdicts: list[SignificanceVerdict] = []
    for i, report in enumerate(sr.tasks.values()):
        rejected = correction.rejected[i]
        adjusted = correction.adjusted[i]
        # A task is REAL only if it survives BOTH the naive test (p<alpha, CI
        # excludes zero, any deflation) AND the multiplicity correction.
        corrected_sig = rejected and report.significant
        verdicts.append(
            verdict_from_report(
                report,
                power=power,
                mei=mei,
                corrected_significant=corrected_sig,
                adjusted_p=adjusted,
                correction_method=method_name,
            )
        )
    return SuiteVerdict(tasks=verdicts, method=method, alpha=alpha, power=power, suite=sr)
