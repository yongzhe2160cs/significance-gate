"""The three-way demo — the whole pitch in one run.

For each of three realistic, lm-eval-shaped scenarios — a real effect, a noise
delta, and an underpowered case — this prints BOTH:

  1. the CLI verdict (what a CI step prints, plus the exit code it would set), and
  2. the rendered Markdown PR comment (what a reviewer would see on the PR).

Then it runs the whole suite through the directory mode to show the multiplicity
correction turning "two tasks look like wins" into "one survives".

Run::

    python examples/make_demo_data.py   # (generates examples/data, committed)
    python examples/demo.py
"""

from __future__ import annotations

from pathlib import Path

import siggate
from siggate.adapters import load_scores, pair_suite
from siggate.comment import render_comment, render_suite_comment
from siggate.config import GateConfig
from siggate.decision import exit_code, failures, suite_exit_code

DATA = Path(__file__).parent / "data"
CFG = GateConfig()  # defaults: alpha 0.05, power 0.80, mei 0.05, standard fail_on

RULE = "═" * 78


def _section(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


def _show(case: str, blurb: str) -> None:
    _section(f"{case.upper()}  —  {blurb}")
    baseline = load_scores(DATA / case / "samples_baseline.jsonl")
    candidate = load_scores(DATA / case / "samples_candidate.jsonl")
    verdict = siggate.compare(baseline, candidate, name=case, **_gate_kwargs())

    print("\n— CLI verdict (`siggate compare ... --format text`) —\n")
    print(verdict.render())

    tripped = failures(verdict, CFG.fail_on)
    code = exit_code(verdict, CFG.fail_on)
    gate_line = "PASS (exit 0)" if code == 0 else f"FAIL (exit {code}) — rules: {tripped}"
    print(f"\n  gate: {gate_line}")

    print("\n— Rendered PR comment (`--format markdown`) —\n")
    print(render_comment(verdict))


def _gate_kwargs() -> dict:
    return {"alpha": CFG.alpha, "power": CFG.power, "mei": CFG.mei}


def main() -> None:
    print("siggate three-way demo — is the reported eval move REAL, NOISE, or UNDERPOWERED?")

    _show("real", "a genuine, well-powered improvement")
    _show("noise", "a tiny delta on a well-powered eval — no real change")
    _show("underpowered", "a promising delta, but far too few samples to call it")

    _section("SUITE  —  4 tasks, multiplicity correction (Holm)")
    suite = pair_suite(DATA / "suite_baseline", DATA / "suite_candidate")
    sv = siggate.compare_suite(suite, alpha=CFG.alpha, power=CFG.power, mei=CFG.mei, method="holm")
    print("\n— CLI verdict —\n")
    print(sv.render())
    code = suite_exit_code(sv, CFG.fail_on)
    print(f"\n  gate: {'PASS (exit 0)' if code == 0 else f'FAIL (exit {code})'}")
    print("\n— Rendered PR comment —\n")
    print(render_suite_comment(sv))

    _section("THE PITCH")
    print(
        "\n  Every tool above shows a number moving between two runs.\n"
        "  siggate is the only one that tells you, in CI, whether the move is real:\n\n"
        "    • REAL          → ship it; the improvement survives the statistics\n"
        "    • NOISE         → you were powered to find an effect; there isn't one\n"
        "    • UNDERPOWERED  → you can't tell yet — here's exactly how many more samples\n"
    )


if __name__ == "__main__":
    main()
