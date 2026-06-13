"""End-to-end CLI: formats, exit codes, suite mode, and error handling."""

from __future__ import annotations

import json

import pytest
from tests.conftest import _paired

from siggate.cli import main


def _write_lmeval(path, scores):
    with path.open("w") as fh:
        for i, s in enumerate(scores):
            fh.write(json.dumps({"doc_id": i, "acc": float(s)}) + "\n")


@pytest.fixture
def real_files(tmp_path):
    base, cand = _paired(1500, 0.70, 0.78, seed=1)
    b = tmp_path / "samples_baseline.jsonl"
    c = tmp_path / "samples_candidate.jsonl"
    _write_lmeval(b, base)
    _write_lmeval(c, cand)
    return b, c


def test_compare_real_passes_default_gate(real_files, capsys):
    b, c = real_files
    code = main(["compare", str(b), str(c)])
    out = capsys.readouterr().out
    assert code == 0  # a real improvement passes the default gate
    assert "REAL" in out


def test_strict_gate_blocks_non_real(tmp_path):
    # A noise comparison under the strict `not-real` rule fails (exit 1).
    base, cand = _paired(1500, 0.72, 0.724, seed=2)
    b = tmp_path / "samples_baseline.jsonl"
    c = tmp_path / "samples_candidate.jsonl"
    _write_lmeval(b, base)
    _write_lmeval(c, cand)
    code = main(["compare", str(b), str(c), "--fail-on", "not-real"])
    assert code == 1  # not REAL -> blocked under the strict rule


def test_no_gate_always_exits_zero(tmp_path):
    base, cand = _paired(1500, 0.72, 0.724, seed=2)
    b = tmp_path / "samples_baseline.jsonl"
    c = tmp_path / "samples_candidate.jsonl"
    _write_lmeval(b, base)
    _write_lmeval(c, cand)
    assert main(["compare", str(b), str(c), "--fail-on", "not-real", "--no-gate"]) == 0


def test_json_format(real_files, capsys):
    b, c = real_files
    main(["compare", str(b), str(c), "--format", "json", "--no-gate"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["label"] == "REAL"
    assert payload["passed"] is True
    assert "summary" in payload
    assert "failures" in payload
    assert payload["gate_enabled"] is False


def test_markdown_format_renders_comment(real_files, capsys):
    b, c = real_files
    main(["compare", str(b), str(c), "--format", "markdown", "--no-gate"])
    out = capsys.readouterr().out
    assert "Significance Gate — REAL" in out
    assert "siggate:significance-gate" in out


def test_output_file_written(real_files, tmp_path, capsys):
    b, c = real_files
    out_md = tmp_path / "comment.md"
    main(["compare", str(b), str(c), "--output", str(out_md), "--no-gate"])
    assert out_md.exists()
    assert "Significance Gate" in out_md.read_text()


def test_suite_mode(tmp_path, capsys):
    a = tmp_path / "a"
    d = tmp_path / "b"
    a.mkdir()
    d.mkdir()
    for task, (ba, ca, seed) in {
        "mmlu": (0.70, 0.78, 10),
        "gsm8k": (0.64, 0.643, 11),
    }.items():
        base, cand = _paired(1200, ba, ca, seed=seed)
        _write_lmeval(a / f"samples_{task}.jsonl", base)
        _write_lmeval(d / f"samples_{task}.jsonl", cand)
    code = main(["compare", str(a), str(d), "--format", "json", "--no-gate"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_tasks"] == 2
    assert "mmlu" in payload["survivors"]
    assert code == 0


def test_mixed_file_and_dir_errors(real_files, tmp_path, capsys):
    b, _ = real_files
    code = main(["compare", str(b), str(tmp_path)])
    assert code == 2
    assert "two files OR two directories" in capsys.readouterr().err


def test_missing_file_errors(tmp_path, capsys):
    code = main(["compare", str(tmp_path / "nope.jsonl"), str(tmp_path / "nope2.jsonl")])
    assert code == 2
    assert "error" in capsys.readouterr().err.lower()


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "siggate" in capsys.readouterr().out
