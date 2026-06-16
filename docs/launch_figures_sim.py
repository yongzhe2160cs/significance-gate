#!/usr/bin/env python3
"""Null-distribution simulation for the launch blog (Experiment 1).

Regenerates the figure marked ``[REGENERATE]`` in
``01-blog-most-eval-wins-are-noise.md`` (Experiment 1: the zero-true-difference
simulation). Two models are, by construction, *exactly equally good* — identical
true accuracy ``p`` on every benchmark. There is no real difference to find. We
then ask the questions the blog cites:

  1. Single benchmark, N items: under the null, how often does B's observed
     score "beat" A's by >= 1pt / >= 2pt?
  2. A 20-benchmark suite: what is the distribution of "benchmarks won" (B's
     point estimate above A's), and how often do you get a >= 12-of-20 headline?
  3. The fix: run the *same* null draws through the SHIPPED gate
     (``siggate.compare`` / ``siggate.compare_suite``) and show how many "wins"
     survive — per-benchmark false-positive rate (~alpha) and suite survivors
     after multiplicity correction (~0).

Parts 1-2 are computed two ways that must agree: an EXACT lattice computation
(convolution of two Binomials — no scipy needed) and a seeded Monte Carlo
cross-check. Part 3 calls the real shipped library so the blog cites the tool's
own behaviour, not a notebook's.

Run:  python docs/launch_figures_sim.py
Seed: fixed (SEED below). Output is deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

SEED = 20260615  # fixed seed -> deterministic output
P_TRUE = 0.50  # identical true accuracy for BOTH models (max-variance null)
N_HEADLINE = 500  # headline per-benchmark sample size cited by the blog
SUITE_K = 20  # benchmarks in the suite
SUITE_N = 500  # items per benchmark in the suite
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Exact lattice math: difference of two independent Binomial(N, p) counts.
# --------------------------------------------------------------------------- #
def _binom_pmf(n: int, p: float) -> np.ndarray:
    """PMF over k = 0..n of Binomial(n, p), via log-gamma for numerical safety."""
    k = np.arange(n + 1)
    log_coeff = (
        math.lgamma(n + 1)
        - np.array([math.lgamma(i + 1) for i in k])
        - np.array([math.lgamma(n - i + 1) for i in k])
    )
    logp = log_coeff + k * math.log(p) + (n - k) * math.log1p(-p)
    return np.exp(logp)


@dataclass
class SingleBenchExact:
    n: int
    p: float
    p_b_strictly_beats_a: float  # P(D > 0)
    p_tie: float  # P(D == 0)
    p_win_ge_1pt: float  # P(B - A >= 1.0 pp), directional
    p_win_ge_2pt: float  # P(B - A >= 2.0 pp), directional
    thresh_1pt_counts: int
    thresh_2pt_counts: int


def single_bench_exact(n: int, p: float) -> SingleBenchExact:
    """Exact null distribution of the observed B-minus-A gap on one benchmark."""
    pmf = _binom_pmf(n, p)
    # Distribution of D = X_B - X_A over d = -n..n (independent => convolve).
    conv = np.convolve(pmf, pmf[::-1])  # index i -> d = i - n
    d_vals = np.arange(-n, n + 1)

    def p_gap_ge(pp: float) -> float:
        # smallest integer count k with k/n*100 >= pp (directional win for B)
        thresh = math.ceil(pp / 100.0 * n - 1e-9)
        return float(conv[d_vals >= thresh].sum())

    t1 = math.ceil(1.0 / 100.0 * n - 1e-9)
    t2 = math.ceil(2.0 / 100.0 * n - 1e-9)
    return SingleBenchExact(
        n=n,
        p=p,
        p_b_strictly_beats_a=float(conv[d_vals > 0].sum()),
        p_tie=float(conv[d_vals == 0].sum()),
        p_win_ge_1pt=p_gap_ge(1.0),
        p_win_ge_2pt=p_gap_ge(2.0),
        thresh_1pt_counts=t1,
        thresh_2pt_counts=t2,
    )


def suite_wins_exact(k: int, n: int, p: float) -> dict:
    """Exact distribution of #benchmarks-won-out-of-k under the null.

    A benchmark is "won" iff B's observed count strictly exceeds A's. The per-
    benchmark win prob is q = P(D>0); #wins ~ Binomial(k, q).
    """
    sb = single_bench_exact(n, p)
    q = sb.p_b_strictly_beats_a
    pmf = _binom_pmf(k, q)
    wins = np.arange(k + 1)
    cdf_ge = {w: float(pmf[wins >= w].sum()) for w in range(k + 1)}
    return {
        "k": k,
        "n_per_bench": n,
        "p": p,
        "q_per_bench_win": q,
        "expected_wins": float((wins * pmf).sum()),
        "pmf": pmf.tolist(),
        "P_ge_10": cdf_ge[10],
        "P_ge_11": cdf_ge[11],
        "P_ge_12": cdf_ge[12],
        "P_ge_13": cdf_ge[13],
        "P_ge_14": cdf_ge[14],
    }


# --------------------------------------------------------------------------- #
# Monte Carlo cross-check (seeded).
# --------------------------------------------------------------------------- #
def montecarlo_checks(n: int, p: float, k: int, suite_n: int, trials: int, rng) -> dict:
    # single benchmark
    xa = rng.binomial(n, p, size=trials)
    xb = rng.binomial(n, p, size=trials)
    gap_pp = (xb - xa) / n * 100.0
    mc_single = {
        "p_b_strictly_beats_a": float((xb > xa).mean()),
        "p_win_ge_1pt": float((gap_pp >= 1.0).mean()),
        "p_win_ge_2pt": float((gap_pp >= 2.0).mean()),
    }
    # suite of k benchmarks per trial
    a_suite = rng.binomial(suite_n, p, size=(trials, k))
    b_suite = rng.binomial(suite_n, p, size=(trials, k))
    wins = (b_suite > a_suite).sum(axis=1)
    mc_suite = {
        "expected_wins": float(wins.mean()),
        "P_ge_12": float((wins >= 12).mean()),
        "P_ge_11": float((wins >= 11).mean()),
        "wins_hist": np.bincount(wins, minlength=k + 1).tolist(),
    }
    return {"trials": trials, "single": mc_single, "suite": mc_suite}


# --------------------------------------------------------------------------- #
# Part 3: drive the SHIPPED gate on the same null.
# --------------------------------------------------------------------------- #
def gate_behaviour(n: int, p: float, k: int, suites: int, per_bench_trials: int, rng) -> dict:
    """False-positive behaviour of the shipped siggate gate under the null.

    Per-benchmark paired draws: same items, identical solve-prob p for both
    models (the null), independent Bernoulli noise. siggate runs its paired
    test; under the null the REAL rate should sit near alpha, and after suite
    multiplicity correction almost nothing should survive.
    """
    import siggate

    # (a) per-benchmark false-positive rate over many independent null comparisons
    real = 0
    labels = {"REAL": 0, "NOISE": 0, "UNDERPOWERED": 0, "INVALID": 0}
    for _ in range(per_bench_trials):
        base = (rng.random(n) < p).astype(float)
        cand = (rng.random(n) < p).astype(float)
        v = siggate.compare(base, cand, name="null", alpha=ALPHA, seed=int(rng.integers(1 << 30)))
        labels[v.label] = labels.get(v.label, 0) + 1
        if v.label == "REAL":
            real += 1

    # (b) suite-level: how many of k benchmarks survive multiplicity correction.
    # We use the shipped tool's OWN definitions: sv.naive_wins (point estimate up)
    # and sv.survivors (REAL after the suite's multiplicity correction).
    survivors = []
    naive_wins = []
    for _ in range(suites):
        suite = {}
        for j in range(k):
            base = (rng.random(n) < p).astype(float)
            cand = (rng.random(n) < p).astype(float)
            suite[f"bench_{j:02d}"] = (base, cand)
        sv = siggate.compare_suite(suite, alpha=ALPHA, method="holm", seed=int(rng.integers(1 << 30)))
        survivors.append(len(sv.survivors))
        naive_wins.append(len(sv.naive_wins))

    survivors = np.array(survivors)
    naive_wins = np.array(naive_wins)
    return {
        "per_bench_trials": per_bench_trials,
        "per_bench_labels": labels,
        "per_bench_false_positive_rate": real / per_bench_trials,
        "suites": suites,
        # NB: the tool's "naive_wins" = tasks won by UNCORRECTED per-task
        # significance (raw p<alpha), NOT point-estimate-up. Distinct from the
        # press-release "point estimate beat the baseline" win in Parts 1-2.
        "naive_significant_wins_mean": float(naive_wins.mean()),
        "naive_significant_wins_max": int(naive_wins.max()),
        "survivors_after_holm_mean": float(survivors.mean()),
        "survivors_after_holm_max": int(survivors.max()),
        "suites_with_zero_survivors_frac": float((survivors == 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-trials", type=int, default=200_000)
    ap.add_argument("--gate-per-bench", type=int, default=2000)
    ap.add_argument("--gate-suites", type=int, default=200)
    ap.add_argument("--json-out", default="docs/launch_figure1_null.json")
    ap.add_argument("--png-out", default="docs/launch_figure1_null.png")
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)

    # Exact single-benchmark table across the sizes the blog references.
    single = {n: asdict(single_bench_exact(n, P_TRUE)) for n in (198, 400, 500)}
    # Robustness: a higher-accuracy benchmark (lower variance) at N=500.
    single_p75 = asdict(single_bench_exact(N_HEADLINE, 0.75))

    suite = suite_wins_exact(SUITE_K, SUITE_N, P_TRUE)
    mc = montecarlo_checks(N_HEADLINE, P_TRUE, SUITE_K, SUITE_N, args.mc_trials, rng)

    gate = None
    if not args.skip_gate:
        gate = gate_behaviour(SUITE_N, P_TRUE, SUITE_K, args.gate_suites, args.gate_per_bench, rng)

    out = {
        "seed": SEED,
        "p_true": P_TRUE,
        "params": {"suite_k": SUITE_K, "suite_n": SUITE_N, "n_headline": N_HEADLINE, "alpha": ALPHA},
        "single_benchmark_exact": single,
        "single_benchmark_p075_exact": single_p75,
        "suite_wins_exact": suite,
        "monte_carlo": mc,
        "shipped_gate": gate,
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console report ----
    print(f"\nNULL SIMULATION  (seed={SEED}, true accuracy p={P_TRUE} for BOTH models)\n")
    print("Single benchmark, exact P(B 'wins' by >= threshold) under zero true difference:")
    print(f"{'N':>6} {'P(B>A)':>9} {'P(tie)':>8} {'P(>=1pt win)':>13} {'P(>=2pt win)':>13}")
    for n, s in single.items():
        print(
            f"{n:>6} {s['p_b_strictly_beats_a']:>9.4f} {s['p_tie']:>8.4f} "
            f"{s['p_win_ge_1pt']:>13.4f} {s['p_win_ge_2pt']:>13.4f}"
        )
    print(
        f"  (robustness, p=0.75, N=500): P(>=1pt win)={single_p75['p_win_ge_1pt']:.4f}  "
        f"P(>=2pt win)={single_p75['p_win_ge_2pt']:.4f}"
    )

    print(f"\nSuite of {SUITE_K} benchmarks (each N={SUITE_N}), exact #-wins distribution:")
    print(f"  per-benchmark win prob q = P(B>A) = {suite['q_per_bench_win']:.4f}")
    print(f"  expected wins = {suite['expected_wins']:.2f} / {SUITE_K}")
    print(f"  P(>=11 wins) = {suite['P_ge_11']:.4f}")
    print(f"  P(>=12 wins) = {suite['P_ge_12']:.4f}   <-- the '12 of 20' headline")
    print(f"  P(>=13 wins) = {suite['P_ge_13']:.4f}")
    print(f"  P(>=14 wins) = {suite['P_ge_14']:.4f}")

    print("\nMonte Carlo cross-check (seeded, n_trials={}):".format(mc["trials"]))
    print(
        f"  single: P(B>A)={mc['single']['p_b_strictly_beats_a']:.4f}  "
        f"P(>=1pt)={mc['single']['p_win_ge_1pt']:.4f}  P(>=2pt)={mc['single']['p_win_ge_2pt']:.4f}"
    )
    print(
        f"  suite : E[wins]={mc['suite']['expected_wins']:.3f}  "
        f"P(>=12)={mc['suite']['P_ge_12']:.4f}  P(>=11)={mc['suite']['P_ge_11']:.4f}"
    )

    if gate is not None:
        print("\nSHIPPED GATE under the same null (siggate.compare / compare_suite):")
        print(
            f"  per-benchmark false-positive (REAL) rate = "
            f"{gate['per_bench_false_positive_rate']:.4f}  "
            f"over {gate['per_bench_trials']} null comparisons  (target ~alpha={ALPHA})"
        )
        print(f"  per-benchmark label counts: {gate['per_bench_labels']}")
        print(
            f"  suite ({gate['suites']} runs of {SUITE_K} benchmarks): "
            f"naive (uncorrected) significant 'wins' mean="
            f"{gate['naive_significant_wins_mean']:.2f} (max {gate['naive_significant_wins_max']}) "
            f"-> survivors after Holm mean={gate['survivors_after_holm_mean']:.3f} "
            f"(max {gate['survivors_after_holm_max']})"
        )
        print(
            f"  {gate['suites_with_zero_survivors_frac'] * 100:.1f}% of suites: "
            f"ZERO benchmarks survive correction"
        )

    # ---- figure: exact null distribution of #wins-out-of-20 ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pmf = np.array(suite["pmf"])
    wins = np.arange(SUITE_K + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#c9442e" if w >= 12 else "#9aa0a6" for w in wins]
    ax.bar(wins, pmf * 100, color=colors, width=0.85)
    ax.set_xlabel("Benchmarks 'won' out of 20 (B's point estimate above A's)")
    ax.set_ylabel("Probability (%)")
    ax.set_title(
        "Null distribution of suite 'wins' — two IDENTICAL models\n"
        f"(true accuracy p={P_TRUE}, N={SUITE_N}/benchmark; P(>=12 wins)="
        f"{suite['P_ge_12']*100:.1f}%)"
    )
    ax.set_xticks(wins)
    ax.axvline(11.5, color="#c9442e", ls="--", lw=1)
    ax.annotate(
        f"'12 of 20' and up\nhappens {suite['P_ge_12']*100:.1f}% of the time\nwith NO real difference",
        xy=(13.2, pmf.max() * 100 * 0.55),
        fontsize=9,
        color="#c9442e",
    )
    fig.tight_layout()
    fig.savefig(args.png_out, dpi=150)
    print(f"\nwrote {args.json_out} and {args.png_out}")


if __name__ == "__main__":
    main()
