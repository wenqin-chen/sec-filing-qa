"""FinanceBench loader on a 3-row synthetic fixture: page +1, both page keys, caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from secqa.core.errors import ConfigError
from secqa.eval import financebench as fb
from secqa.eval.financebench import (
    download_pdfs,
    load_financebench,
    load_questions_jsonl,
    question_from_row,
    questions_from_rows,
    save_questions_jsonl,
)
from tests.eval.conftest import FB_MINI, FB_ROWS


def _rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FB_ROWS.read_text(encoding="utf-8").splitlines() if line]


def test_rows_convert_with_one_based_pages() -> None:
    questions = questions_from_rows(_rows())
    assert [q.id for q in questions] == ["synthetic_row_1", "synthetic_row_2", "synthetic_row_3"]
    first, second, third = questions
    assert first.evidence[0].page_num == 1  # evidence_page_num 0 -> page 1
    assert second.evidence[0].page_num == 2  # page_number 1 -> page 2 (alternate key)
    assert [e.page_num for e in third.evidence] == [2, 1]
    assert first.doc_period == "2023" and third.doc_period == "2022"  # int and str both -> str
    assert first.doc_link and second.doc_link is None
    assert first.evidence[0].text.startswith("Total net sales")


def test_row_validation_errors() -> None:
    row = _rows()[0]
    with pytest.raises(ValueError, match="missing"):
        question_from_row({**row, "answer": ""})
    bad_page = {**row, "evidence": [{"evidence_text": "x", "doc_name": row["doc_name"]}]}
    with pytest.raises(ValueError, match="no page number"):
        question_from_row(bad_page)
    negative = {**row, "evidence": [{"evidence_text": "x", "evidence_page_num": -1}]}
    with pytest.raises(ValueError, match=">= 0"):
        question_from_row(negative)
    with pytest.raises(ValueError, match="duplicate"):
        questions_from_rows([row, row])


def test_jsonl_round_trip_and_fixture(tmp_path: Path) -> None:
    questions = questions_from_rows(_rows())
    path = tmp_path / "q.jsonl"
    assert save_questions_jsonl(questions, path) == 3
    assert load_questions_jsonl(path) == questions
    mini = load_questions_jsonl(FB_MINI)
    assert len(mini) == 6
    assert all(e.page_num >= 1 for q in mini for e in q.evidence)
    with pytest.raises(ConfigError, match="not found"):
        load_questions_jsonl(tmp_path / "missing.jsonl")
    path.write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid question record"):
        load_questions_jsonl(path)


def test_load_financebench_uses_cache_and_records_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_rows(split: str, revision: str | None, cache_dir: Path) -> tuple[list, dict]:
        calls.append((split, revision))
        return _rows(), {
            "dataset": fb.DATASET_NAME,
            "split": split,
            "revision": revision or "main",
            "fingerprint": "fp",
            "n_rows": 3,
            "columns": [],
        }

    monkeypatch.setattr(fb, "_load_hf_rows", fake_rows)
    first = load_financebench(cache_dir=tmp_path, revision="abc")
    assert len(first) == 3 and calls == [("train", "abc")]
    assert (tmp_path / "financebench_train.jsonl").is_file()
    info = json.loads((tmp_path / fb.DATASET_INFO_NAME).read_text(encoding="utf-8"))
    assert info["revision"] == "abc" and info["n_rows"] == 3 and "loaded_at" in info
    assert fb.dataset_revision(tmp_path) == "abc"
    # cached: no second download for the same (or unspecified) revision
    assert load_financebench(cache_dir=tmp_path) == first
    assert load_financebench(cache_dir=tmp_path, revision="abc") == first
    assert calls == [("train", "abc")]
    # a different revision re-downloads
    load_financebench(cache_dir=tmp_path, revision="def")
    assert calls[-1] == ("train", "def")


def test_load_financebench_wraps_download_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(split: str, revision: str | None, cache_dir: Path) -> tuple[list, dict]:
        raise RuntimeError("no network")

    monkeypatch.setattr(fb, "_load_hf_rows", boom)
    with pytest.raises(ConfigError, match="no network"):
        load_financebench(cache_dir=tmp_path)


def test_download_pdfs_delegates_with_doc_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_download(doc_names: list[str], out_dir: Path, edgar: Any, doc_links: Any) -> str:
        captured.update(doc_names=list(doc_names), out_dir=out_dir, links=doc_links)
        return "report"

    monkeypatch.setattr(fb, "download_financebench_pdfs", fake_download)
    questions = questions_from_rows(_rows())
    assert download_pdfs(questions, tmp_path / "pdfs") == "report"
    assert captured["doc_names"] == ["FIXTURE_2023_10K", "OTHER_2022_10K"]  # de-duplicated
    assert captured["links"] == {"FIXTURE_2023_10K": "https://example.invalid/fixture.pdf"}


@pytest.mark.live
def test_live_financebench_has_150_rows(tmp_path: Path) -> None:
    questions = load_financebench(cache_dir=tmp_path)
    assert len(questions) == 150
    assert all(e.page_num >= 1 for q in questions for e in q.evidence)
