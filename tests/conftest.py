"""Shared fixtures: small, deterministic paired score vectors for each label."""

from __future__ import annotations

import numpy as np
import pytest


def _paired(n: int, base_acc: float, cand_acc: float, seed: int, corr: float = 0.8):
    """Two correlated 0/1 vectors via a shared latent difficulty (paired)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.0, n)
    nb = rng.normal(0, np.sqrt(1 - corr**2) + 0.3, n)
    nc = rng.normal(0, np.sqrt(1 - corr**2) + 0.3, n)
    from math import erf, sqrt

    def q(p):
        lo, hi = -8.0, 8.0
        for _ in range(80):
            m = 0.5 * (lo + hi)
            if 0.5 * (1 + erf(m / sqrt(2))) < p:
                lo = m
            else:
                hi = m
        return 0.5 * (lo + hi)

    base = ((q(base_acc) * 1.1 - z + nb) > 0).astype(float).tolist()
    cand = ((q(cand_acc) * 1.1 - z + nc) > 0).astype(float).tolist()
    return base, cand


@pytest.fixture
def real_case():
    # Big, clearly-separated, well-powered -> REAL.
    return _paired(1500, 0.70, 0.78, seed=1)


@pytest.fixture
def noise_case():
    # Big n, ~zero delta, well-powered -> NOISE.
    return _paired(1500, 0.72, 0.722, seed=2)


@pytest.fixture
def underpowered_case():
    # Small n -> can't resolve a meaningful effect -> UNDERPOWERED.
    return _paired(80, 0.66, 0.70, seed=5, corr=0.5)
