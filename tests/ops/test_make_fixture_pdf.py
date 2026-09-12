"""scripts/make_fixture_pdf.py renders the synthetic corpus so pypdfium2 reads it back verbatim."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from secqa.core.errors import ConfigError
from secqa.ingest import extract_pdf_pages
from tests.ops.conftest import FIXTURES

PAGES = FIXTURES / "eval_fixture_pages.json"


@pytest.fixture
def mfp(script: Callable[[str], ModuleType]) -> ModuleType:
    return script("make_fixture_pdf")


def test_renders_every_document_and_text_round_trips(mfp: ModuleType, tmp_path: Path) -> None:
    written = mfp.render_all(PAGES, tmp_path)
    corpus = json.loads(PAGES.read_text(encoding="utf-8"))
    assert sorted(p.stem for p in written) == sorted(corpus)
    for path in written:
        pages = extract_pdf_pages(path, path.stem)
        expected = corpus[path.stem]["pages"]
        assert len(pages) == len(expected)
        for page, text in zip(pages, expected, strict=True):
            assert " ".join(text.split()) == page.text


def test_doc_filter_and_unknown_doc(mfp: ModuleType, tmp_path: Path) -> None:
    written = mfp.render_all(PAGES, tmp_path, only=["FIXTURE_2023_10K"])
    assert [p.name for p in written] == ["FIXTURE_2023_10K.pdf"]
    with pytest.raises(ConfigError, match="unknown documents"):
        mfp.render_all(PAGES, tmp_path, only=["NOPE"])


def test_main_exit_codes(
    mfp: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert mfp.main(["--pages-json", str(PAGES), "--out-dir", str(tmp_path)]) == 0
    assert "FIXTURE_2023_10K.pdf" in capsys.readouterr().out
    assert (
        mfp.main(["--pages-json", str(tmp_path / "missing.json"), "--out-dir", str(tmp_path)]) == 2
    )


def test_write_pdf_requires_a_page(mfp: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        mfp.write_pdf([], tmp_path / "x.pdf")
