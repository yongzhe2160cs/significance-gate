"""Generate the three-way demo data: a real win, a noise delta, an underpowered case.

Writes lm-evaluation-harness ``--log_samples``-shaped JSONL files (one record per
line, ``{"doc_id": i, "acc": 0/1, ...}``) for a baseline and a candidate run of
each scenario, under ``examples/data/``. Fully deterministic (seeded), so the
committed files and this script always agree — re-run to regenerate.

The per-sample scores are *paired*: a shared latent difficulty makes both models
right/wrong on the same items (the real-world correlation that makes a paired
analysis the correct one). That correlation is exactly what a naive unpaired
eyeball misses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"


def _paired_runs(
    rng: np.random.Generator,
    n: int,
    base_acc: float,
    cand_acc: float,
    difficulty_sd: float = 1.0,
) -> tuple[list[int], list[int]]:
    """Two correlated 0/1 score vectors via a shared per-item latent difficulty.

    Each item has a latent difficulty ``z_i``; a model passes when its ability
    plus item noise clears the difficulty. Both models share ``z_i`` (so they
    tend to miss the same hard items), which is what makes the runs paired.
    """
    from math import erf, sqrt

    def _phi_inv(p: float) -> float:
        # ability offset so that P(pass) ~= target accuracy under the model below
        lo, hi = -10.0, 10.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            cdf = 0.5 * (1 + erf(mid / sqrt(2)))
            if cdf < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    z = rng.normal(0.0, difficulty_sd, size=n)  # shared item difficulty
    item_noise_b = rng.normal(0.0, 0.5, size=n)
    item_noise_c = rng.normal(0.0, 0.5, size=n)
    a_base = _phi_inv(base_acc) * sqrt(difficulty_sd**2 + 0.25)
    a_cand = _phi_inv(cand_acc) * sqrt(difficulty_sd**2 + 0.25)
    baseline = ((a_base - z + item_noise_b) > 0).astype(int)
    candidate = ((a_cand - z + item_noise_c) > 0).astype(int)
    return baseline.tolist(), candidate.tolist()


def _write(path: Path, scores: list[int], task: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for doc_id, acc in enumerate(scores):
            fh.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "task": task,
                        "acc": float(acc),
                        "target": "synthetic",
                    }
                )
                + "\n"
            )


def main() -> None:
    # Each scenario gets its own seeded RNG so its draw is reproducible
    # regardless of generation order.

    # (a) REAL effect: a clear, well-powered win. 70.0% -> ~75% on n=1500.
    rng = np.random.default_rng(1)
    base, cand = _paired_runs(rng, n=1500, base_acc=0.700, cand_acc=0.745, difficulty_sd=1.0)
    _write(DATA / "real" / "samples_baseline.jsonl", base, "mmlu")
    _write(DATA / "real" / "samples_candidate.jsonl", cand, "mmlu")

    # (b) NOISE: well-powered (n=1500) but the models are essentially the same.
    #     The delta is ~zero and the eval CAN detect a meaningful effect -> noise.
    rng = np.random.default_rng(2)
    base, cand = _paired_runs(rng, n=1500, base_acc=0.720, cand_acc=0.723, difficulty_sd=1.0)
    _write(DATA / "noise" / "samples_baseline.jsonl", base, "gsm8k")
    _write(DATA / "noise" / "samples_candidate.jsonl", cand, "gsm8k")

    # (c) UNDERPOWERED: a promising-looking +3% on far too few samples (n=180)
    #     to resolve — the "you'd need ~N more samples" headline case.
    rng = np.random.default_rng(3)
    base, cand = _paired_runs(rng, n=180, base_acc=0.680, cand_acc=0.710, difficulty_sd=0.8)
    _write(DATA / "underpowered" / "samples_baseline.jsonl", base, "arc_challenge")
    _write(DATA / "underpowered" / "samples_candidate.jsonl", cand, "arc_challenge")

    # A small suite (for the directory / multiplicity demo): 4 tasks, only one
    # a genuine win; the rest noise. Naively a couple look like wins; the
    # multiplicity correction culls the lucky ones.
    suite_specs = {
        "mmlu": (0.700, 0.742, 1200),  # real win
        "gsm8k": (0.640, 0.648, 1200),  # noise
        "arc_challenge": (0.810, 0.816, 1200),  # noise
        "hellaswag": (0.880, 0.892, 1200),  # lucky-looking near-miss
    }
    for i, (task, (b_acc, c_acc, n)) in enumerate(suite_specs.items()):
        rng = np.random.default_rng(100 + i)
        base, cand = _paired_runs(rng, n=n, base_acc=b_acc, cand_acc=c_acc, difficulty_sd=1.0)
        _write(DATA / "suite_baseline" / f"samples_{task}.jsonl", base, task)
        _write(DATA / "suite_candidate" / f"samples_{task}.jsonl", cand, task)

    print(f"wrote demo data under {DATA}")


if __name__ == "__main__":
    main()
