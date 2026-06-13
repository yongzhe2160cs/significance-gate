"""Loading eval outputs into ``{sample_id: score}`` — pluggable, with auto-detect.

The actual parsers are deltagate's (``LMEvalHarnessAdapter``, ``InspectLogAdapter``,
``RawScoresAdapter``); siggate adds a name/extension registry, file-type
auto-detection, and directory pairing for whole suites. Register a new adapter
with :func:`register_adapter` without touching the core.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from deltagate.adapters import (
    InspectLogAdapter,
    LMEvalHarnessAdapter,
    RawScoresAdapter,
    ScoresAdapter,
)

__all__ = [
    "build_adapter",
    "register_adapter",
    "detect_adapter",
    "load_scores",
    "pair_suite",
    "ADAPTERS",
]

# name -> factory(metric) -> ScoresAdapter. `metric` is the per-sample metric key
# (lm-eval) or scorer name (Inspect); ignored by the raw adapter.
ADAPTERS: dict[str, Callable[[str], ScoresAdapter]] = {
    "lm-eval": lambda metric: LMEvalHarnessAdapter(metric=metric or "acc"),
    "inspect": lambda metric: InspectLogAdapter(scorer=metric or None),
    "raw": lambda metric: RawScoresAdapter(),
}

# Extension -> adapter name, for auto-detection.
_EXT_MAP = {
    ".jsonl": "lm-eval",
    ".eval": "inspect",
    ".json": "raw",
    ".csv": "raw",
}


def register_adapter(name: str, factory: Callable[[str], ScoresAdapter]) -> None:
    """Register a custom adapter factory under ``name`` (usable via ``--adapter``)."""
    ADAPTERS[name] = factory


def detect_adapter(source: str | Path) -> str:
    """Pick an adapter name from a file's extension.

    ``.jsonl`` -> lm-eval, ``.eval`` -> inspect, ``.json``/``.csv`` -> raw.
    """
    suffix = Path(source).suffix.lower()
    name = _EXT_MAP.get(suffix)
    if name is None:
        raise ValueError(
            f"cannot auto-detect adapter for {Path(source).name!r} "
            f"(extension {suffix!r}); pass --adapter explicitly "
            f"(one of {sorted(ADAPTERS)})"
        )
    return name


def build_adapter(name: str, metric: str = "acc") -> ScoresAdapter:
    """Instantiate a registered adapter by ``name`` with the given ``metric`` key."""
    if name not in ADAPTERS:
        raise ValueError(f"unknown adapter {name!r}; available: {sorted(ADAPTERS)}")
    return ADAPTERS[name](metric)


def load_scores(
    source: str | Path, adapter: str = "auto", metric: str = "acc"
) -> Mapping[object, float]:
    """Load one run's ``{sample_id: score}`` from ``source``.

    ``adapter="auto"`` detects from the file extension; otherwise the named
    adapter is used.
    """
    name = detect_adapter(source) if adapter == "auto" else adapter
    return build_adapter(name, metric=metric).load(source)


def pair_suite(
    dir_a: str | Path, dir_b: str | Path, adapter: str = "auto", metric: str = "acc"
) -> dict[str, tuple[Mapping[object, float], Mapping[object, float]]]:
    """Pair eval files across two directories by filename stem into a suite.

    Each file present in *both* directories (matched by stem, e.g.
    ``samples_mmlu.jsonl``) becomes one task ``{stem: (scores_a, scores_b)}``.
    """
    a_root, b_root = Path(dir_a), Path(dir_b)
    a_files = {
        p.stem: p for p in sorted(a_root.iterdir()) if p.is_file() and not p.name.startswith(".")
    }
    b_files = {
        p.stem: p for p in sorted(b_root.iterdir()) if p.is_file() and not p.name.startswith(".")
    }
    common = sorted(set(a_files) & set(b_files))
    if not common:
        raise ValueError(
            f"no task files share a name between {a_root} and {b_root} "
            f"(A has {sorted(a_files)}, B has {sorted(b_files)})"
        )
    suite: dict[str, tuple[Mapping[object, float], Mapping[object, float]]] = {}
    for stem in common:
        task = stem.replace("samples_", "")
        suite[task] = (
            load_scores(a_files[stem], adapter=adapter, metric=metric),
            load_scores(b_files[stem], adapter=adapter, metric=metric),
        )
    return suite
