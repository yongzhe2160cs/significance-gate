"""Render a verdict as a clean Markdown PR comment.

The comment is the product's face: a reviewer should see, at a glance, whether
the reported eval move is real (🟢), within noise (⚪), or simply unresolved at
this sample size (🟡) — with the numbers and the "N more samples needed" line.

A stable HTML marker (:data:`COMMENT_MARKER`) is embedded so a CI job can find
and *update* its previous comment instead of posting a new one each run.
"""

from __future__ import annotations

from .gate import REAL, UNDERPOWERED, SignificanceVerdict, SuiteVerdict, _fmt

__all__ = ["COMMENT_MARKER", "render_comment", "render_suite_comment", "render_markdown"]

COMMENT_MARKER = "<!-- siggate:significance-gate -->"
_FOOTER = (
    "<sub>🟢 real · 🟡 underpowered · ⚪ noise · 🔴 invalid — "
    "[significance-gate](https://github.com/yongzhe2160cs/significance-gate) "
    "· powered by [deltagate](https://github.com/yongzhe2160cs/eval-reliability)</sub>"
)


def _pct(v: float, proportion: bool, *, signed: bool = True) -> str:
    return _fmt(v, proportion, signed=signed)


def render_comment(verdict: SignificanceVerdict) -> str:
    """Render a single-comparison verdict as a Markdown comment body."""
    v = verdict
    lines = [
        COMMENT_MARKER,
        f"### {v.emoji} Significance Gate — {v.label}",
        "",
        f"**`{v.name}`** — {v.summary().split(': ', 1)[-1]}",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| delta (candidate − baseline) | **{_pct(v.delta, v.proportion)}** |",
        f"| baseline → candidate | {_pct(v.baseline_mean, v.proportion, signed=False)} → "
        f"{_pct(v.candidate_mean, v.proportion, signed=False)} |",
        f"| {int(round((1 - v.alpha) * 100))}% CI | [{_pct(v.ci_lower, v.proportion)}, "
        f"{_pct(v.ci_upper, v.proportion)}] |",
        f"| p-value | {v.p_value:.3g} |",
        f"| samples (n) | {v.n:,} |",
        f"| min detectable @ n={v.n} | {_pct(v.min_detectable, v.proportion, signed=False)} |",
        f"| P(real) | {v.prob_real:.2f} |",
    ]
    if v.deflated is not None:
        lines.append(f"| deflated (best-of-{v.n_trials}) | {v.deflated:.2f} |")
    if v.corrected and v.adjusted_p is not None:
        lines.append(f"| {v.correction_method} adjusted p | {v.adjusted_p:.3g} |")
    lines.append("")
    lines.append(_verdict_callout(v))
    if v.flags:
        lines.append("")
        for name, message in v.flags:
            lines.append(f"> ⚠️ **{name}** — {message}")
    lines.append("")
    lines.append(_FOOTER)
    return "\n".join(lines)


def _verdict_callout(v: SignificanceVerdict) -> str:
    if v.label == REAL:
        return (
            f"🟢 **Real change.** The candidate moved {_pct(v.delta, v.proportion)} and the "
            f"effect is statistically supported at α={v.alpha:g}"
            + (
                f", surviving {v.correction_method} multiplicity correction."
                if v.corrected
                else "."
            )
        )
    if v.label == UNDERPOWERED:
        if v.n_more:
            return (
                f"🟡 **Underpowered.** This delta is below what n={v.n:,} can resolve. "
                f"You'd need ~**{v.n_more:,} more samples** ({v.n_needed:,} total) "
                f"to detect an effect this size at {v.power:.0%} power."
            )
        return f"🟡 **Underpowered.** This delta is below the minimum detectable at n={v.n:,}."
    if v.label == "INVALID":
        return "🔴 **Invalid comparison.** See the flag above — there is nothing to gate."
    # NOISE
    return (
        f"⚪ **Within noise.** {_pct(v.delta, v.proportion)} is not distinguishable from "
        f"zero (p={v.p_value:.3g}) at n={v.n:,}. Report it as no change."
    )


def render_suite_comment(suite: SuiteVerdict) -> str:
    """Render a suite verdict as a Markdown comment: one row per task + survivors."""
    surv = suite.survivors
    header_emoji = "🟢" if surv else "⚪"
    lines = [
        COMMENT_MARKER,
        f"### {header_emoji} Significance Gate — suite ({len(suite.tasks)} tasks)",
        "",
        f"After **{suite.method.upper()}** multiplicity correction, "
        f"**{len(surv)} of {len(suite.tasks)}** reported changes are real "
        f"(naive per-task wins: {len(suite.naive_wins)}).",
        "",
        "| task | delta | p | adj p | n | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for t in suite.tasks:
        adj = f"{t.adjusted_p:.3g}" if t.adjusted_p is not None else "—"
        lines.append(
            f"| `{t.name}` | {_pct(t.delta, t.proportion)} | {t.p_value:.3g} | "
            f"{adj} | {t.n:,} | {t.emoji} {t.label} |"
        )
    lines.append("")
    if surv:
        lines.append("🟢 **Survives correction:** " + ", ".join(f"`{t.name}`" for t in surv))
    else:
        lines.append(
            "⚪ **No task survives correction.** Every reported win is within noise "
            "once multiple comparisons are accounted for."
        )
    lines.append("")
    lines.append(_FOOTER)
    return "\n".join(lines)


def render_markdown(verdict: SignificanceVerdict | SuiteVerdict) -> str:
    """Dispatch to the right renderer for a single or suite verdict."""
    if isinstance(verdict, SuiteVerdict):
        return render_suite_comment(verdict)
    return render_comment(verdict)
