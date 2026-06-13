"""``siggate`` command-line interface — the CI-check entry point.

    siggate compare BASELINE CANDIDATE [options]

``BASELINE`` and ``CANDIDATE`` are either two eval files (one comparison) or two
directories (a suite, paired by filename). The verdict is printed in the chosen
format and the process exit code reflects the gate, so it drops straight into a
CI step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .adapters import load_scores, pair_suite
from .comment import render_markdown
from .config import FAIL_RULES, GateConfig, load_config
from .decision import exit_code, failures, suite_exit_code, suite_failures
from .gate import SignificanceVerdict, SuiteVerdict, compare, compare_suite

FORMATS = ("text", "json", "markdown", "github")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siggate",
        description=(
            "The vendor-neutral significance gate for AI evals: is a reported "
            "eval improvement statistically REAL, or within noise?"
        ),
    )
    parser.add_argument("--version", action="version", version=f"siggate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    cmp_ = sub.add_parser(
        "compare",
        help="gate a candidate eval run against a baseline (files or directories)",
        description=(
            "Compare two eval runs on the same samples and report whether the "
            "candidate's change over the baseline is real. Pass two files for a "
            "single comparison, or two directories for a multi-task suite."
        ),
    )
    cmp_.add_argument("baseline", help="baseline eval output (file) or directory of outputs")
    cmp_.add_argument("candidate", help="candidate eval output (file) or directory of outputs")
    cmp_.add_argument("--config", type=Path, default=None, help="path to siggate.toml")
    cmp_.add_argument(
        "--adapter",
        default=None,
        help="auto | lm-eval | inspect | raw  (default: auto-detect by extension)",
    )
    cmp_.add_argument(
        "--metric",
        default=None,
        help="per-sample metric key (lm-eval, e.g. acc/acc_norm) or scorer name (Inspect)",
    )
    cmp_.add_argument("--alpha", type=float, default=None, help="significance level (default 0.05)")
    cmp_.add_argument("--power", type=float, default=None, help="target power (default 0.80)")
    cmp_.add_argument(
        "--mei",
        type=float,
        default=None,
        help="minimum effect of interest in metric units (default 0.05); "
        "splits NOISE (powered, no effect) from UNDERPOWERED (can't resolve it)",
    )
    cmp_.add_argument(
        "--n-trials",
        type=int,
        default=None,
        dest="n_trials",
        help="best-of-N variants tried before reporting this run (selection correction)",
    )
    cmp_.add_argument(
        "--correction",
        default=None,
        choices=("holm", "bh"),
        help="suite multiplicity correction (default holm)",
    )
    cmp_.add_argument(
        "--fail-on",
        default=None,
        help=(
            "comma-separated gate rules that fail the check; "
            f"choices: {','.join(sorted(FAIL_RULES))}. Use 'none' to report only."
        ),
    )
    cmp_.add_argument("--name", default=None, help="label for the comparison (single-file mode)")
    cmp_.add_argument(
        "--format",
        default="text",
        choices=FORMATS,
        help="text | json | markdown | github (default text)",
    )
    cmp_.add_argument(
        "--output",
        type=Path,
        default=None,
        help="also write the rendered Markdown comment to this file (for PR posting)",
    )
    cmp_.add_argument(
        "--no-gate",
        action="store_true",
        help="always exit 0 (report only; do not let the verdict fail the build)",
    )
    cmp_.add_argument("--quiet", action="store_true", help="suppress the report on stdout")
    return parser


def _resolve_config(args: argparse.Namespace) -> GateConfig:
    base = load_config(args.config)
    fail_on = None
    if args.fail_on is not None:
        raw = [r.strip() for r in args.fail_on.split(",") if r.strip()]
        fail_on = [] if raw == ["none"] else raw
    return base.merged(
        alpha=args.alpha,
        power=args.power,
        mei=args.mei,
        adapter=args.adapter,
        metric=args.metric,
        correction=args.correction,
        n_trials=args.n_trials,
        fail_on=fail_on,
    )


def _emit(
    verdict: SignificanceVerdict | SuiteVerdict,
    fmt: str,
    cfg: GateConfig,
    gated: bool,
    *,
    quiet: bool,
) -> None:
    if quiet:
        return
    if fmt in ("markdown", "github"):
        print(render_markdown(verdict))
        return
    if fmt == "json":
        payload = verdict.to_dict()
        if isinstance(verdict, SuiteVerdict):
            payload["failures"] = suite_failures(verdict, cfg.fail_on)
        else:
            payload["failures"] = failures(verdict, cfg.fail_on)
        payload["gate_enabled"] = gated
        print(json.dumps(payload, indent=2))
        return
    # text
    print(verdict.render())


def _run_compare(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    baseline, candidate = Path(args.baseline), Path(args.candidate)
    gated = not args.no_gate

    if baseline.is_dir() and candidate.is_dir():
        suite = pair_suite(baseline, candidate, adapter=cfg.adapter, metric=cfg.metric)
        verdict: SignificanceVerdict | SuiteVerdict = compare_suite(
            suite, alpha=cfg.alpha, power=cfg.power, mei=cfg.mei, method=cfg.correction
        )
        code = suite_exit_code(verdict, cfg.fail_on)
    elif baseline.is_dir() or candidate.is_dir():
        raise ValueError("pass two files OR two directories (a suite), not a mix of both")
    else:
        scores_a = load_scores(baseline, adapter=cfg.adapter, metric=cfg.metric)
        scores_b = load_scores(candidate, adapter=cfg.adapter, metric=cfg.metric)
        name = args.name or candidate.stem.replace("samples_", "")
        verdict = compare(
            scores_a,
            scores_b,
            name=name,
            alpha=cfg.alpha,
            power=cfg.power,
            mei=cfg.mei,
            n_trials=cfg.n_trials,
        )
        code = exit_code(verdict, cfg.fail_on)

    _emit(verdict, args.format, cfg, gated, quiet=args.quiet)

    if args.output is not None:
        args.output.write_text(render_markdown(verdict))

    return code if gated else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compare":
        try:
            return _run_compare(args)
        except SystemExit:
            raise
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            print(f"siggate: error: {exc}", file=sys.stderr)
            return 2
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
