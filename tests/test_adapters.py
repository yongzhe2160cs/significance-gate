"""Adapter registry, auto-detection, score loading, and suite pairing."""

from __future__ import annotations

import json

import pytest

from siggate.adapters import (
    ADAPTERS,
    build_adapter,
    detect_adapter,
    load_scores,
    pair_suite,
    register_adapter,
)


def _write_lmeval(path, scores, metric="acc"):
    with path.open("w") as fh:
        for i, s in enumerate(scores):
            fh.write(json.dumps({"doc_id": i, metric: float(s)}) + "\n")


def test_detect_adapter_by_extension():
    assert detect_adapter("x.jsonl") == "lm-eval"
    assert detect_adapter("x.eval") == "inspect"
    assert detect_adapter("x.json") == "raw"
    assert detect_adapter("x.csv") == "raw"


def test_detect_adapter_unknown_extension():
    with pytest.raises(ValueError, match="auto-detect"):
        detect_adapter("scores.txt")


def test_load_lmeval_jsonl(tmp_path):
    p = tmp_path / "samples_mmlu.jsonl"
    _write_lmeval(p, [1, 0, 1, 1])
    scores = load_scores(p)  # auto -> lm-eval
    assert scores == {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0}


def test_load_raw_csv(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("id,score\na,1\nb,0\nc,1\n")
    assert load_scores(p) == {"a": 1.0, "b": 0.0, "c": 1.0}


def test_load_raw_json_mapping(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"q1": 1, "q2": 0}))
    assert load_scores(p) == {"q1": 1.0, "q2": 0.0}


def test_metric_selection_for_lmeval(tmp_path):
    p = tmp_path / "s.jsonl"
    with p.open("w") as fh:
        for i in range(3):
            fh.write(json.dumps({"doc_id": i, "acc": 1.0, "acc_norm": 0.0}) + "\n")
    assert load_scores(p, metric="acc") == {0: 1.0, 1: 1.0, 2: 1.0}
    assert load_scores(p, metric="acc_norm") == {0: 0.0, 1: 0.0, 2: 0.0}


def test_build_adapter_unknown():
    with pytest.raises(ValueError, match="unknown adapter"):
        build_adapter("nope")


def test_register_custom_adapter(tmp_path):
    class Const:
        def load(self, source):
            return {0: 1.0, 1: 1.0}

    register_adapter("const", lambda metric: Const())
    try:
        assert load_scores("ignored", adapter="const") == {0: 1.0, 1: 1.0}
    finally:
        ADAPTERS.pop("const", None)


def test_pair_suite(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_lmeval(a / "samples_mmlu.jsonl", [1, 0, 1])
    _write_lmeval(b / "samples_mmlu.jsonl", [1, 1, 1])
    _write_lmeval(a / "samples_gsm8k.jsonl", [0, 0, 1])
    _write_lmeval(b / "samples_gsm8k.jsonl", [0, 1, 1])
    # A file present only on one side is ignored.
    _write_lmeval(a / "samples_only_a.jsonl", [1, 1, 1])
    suite = pair_suite(a, b)
    assert set(suite) == {"mmlu", "gsm8k"}
    assert suite["mmlu"][0] == {0: 1.0, 1: 0.0, 2: 1.0}


def test_pair_suite_no_common_files(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_lmeval(a / "samples_x.jsonl", [1])
    _write_lmeval(b / "samples_y.jsonl", [1])
    with pytest.raises(ValueError, match="no task files share a name"):
        pair_suite(a, b)
