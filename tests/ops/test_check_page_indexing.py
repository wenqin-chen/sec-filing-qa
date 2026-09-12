"""scripts/check_page_indexing.py on a reportlab PDF and synthetic questions (no dataset text)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from secqa.core.contracts import Evidence, FBQuestion
from secqa.ingest import extract_pdf_pages

DOC = "FIXTURE_2023_10K"


@pytest.fixture
def cpi(script: Callable[[str], ModuleType]) -> ModuleType:
    return script("check_page_indexing")


@pytest.fixture
def pages(fixture_pdf: Path) -> dict[int, str]:
    return {p.page_num: p.text for p in extract_pdf_pages(fixture_pdf, DOC)}


def _question(qid: str, page: int, text: str, doc: str = DOC) -> FBQuestion:
    return FBQuestion(
        id=qid,
        company="Fixture Corp",
        doc_name=doc,
        question_type="metrics-generated",
        question="synthetic question",
        answer="synthetic answer",
        justification="",
        evidence=[Evidence(doc_name=doc, page_num=page, text=text)],
    )


def test_normalise_is_whitespace_and_punctuation_tolerant(cpi: ModuleType) -> None:
    assert cpi.normalise("Total  net\nsales were $1,577 million") == cpi.normalise(
        "total net sales were 1 577 million"
    )


def test_locate_evidence_reports_offsets(cpi: ModuleType, pages: dict[int, str]) -> None:
    text = "Operating income was $245 million and net income was $190 million"
    assert cpi.locate_evidence(text, pages, gold_page=2) == (0,)
    assert cpi.locate_evidence(text, pages, gold_page=1) == (1,)  # found one page later
    assert cpi.locate_evidence(text, pages, gold_page=3) == (-1,)
    assert cpi.locate_evidence("not in the document at all", pages, gold_page=2) == ()


def test_check_question_pass_fail_and_skip(cpi: ModuleType, pages: dict[int, str]) -> None:
    by_doc = {DOC: pages}
    good = cpi.check_question(_question("q1", 2, "Operating income was $245 million"), by_doc)
    assert good.checkable and good.passed
    off_by_one = cpi.check_question(_question("q2", 3, "Operating income was $245 million"), by_doc)
    assert off_by_one.checkable and not off_by_one.passed
    assert off_by_one.hits[0].found_offsets == (-1,)
    short = cpi.check_question(_question("q3", 1, "$1,577"), by_doc)
    assert not short.checkable  # evidence too short to be meaningful
    missing_doc = cpi.check_question(
        _question("q4", 1, "Operating income was $245 million", "X"), by_doc
    )
    assert missing_doc.error and not missing_doc.checkable


def test_run_and_report_on_fixture_pdf(cpi: ModuleType, fixture_pdf: Path, tmp_path: Path) -> None:
    questions = [
        _question("q1", 1, "Total net sales were $1,577 million in fiscal 2023"),
        _question("q2", 2, "Operating income was $245 million"),
        _question("q3", 3, "Cash and cash equivalents were $410 million"),
        _question("q4", 2, "Cash and cash equivalents were $410 million"),  # gold page wrong (+1)
    ]
    report = cpi.run(questions, fixture_pdf.parent, n=25, seed=0, min_pass=0.9)
    assert len(report.results) == 4
    assert report.n_passed == 3 and report.pass_rate == pytest.approx(0.75)
    assert not report.ok
    assert report.offset_histogram() == {"+1": 1}

    block = cpi.render_report(report, now=datetime(2026, 9, 11, tzinfo=UTC))
    assert block.startswith(cpi.START_MARKER) and block.rstrip().endswith(cpi.END_MARKER)
    assert "**FAIL**" in block and "| +1 | 1 |" in block and "`q4`" in block
    assert "Total net sales" not in block, "report must not carry evidence text"

    out = tmp_path / "DATA.md"
    out.write_text("# Data\n\nintro\n", encoding="utf-8")
    cpi.update_markdown(out, block)
    first = out.read_text(encoding="utf-8")
    assert first.startswith("# Data\n\nintro\n") and first.count(cpi.START_MARKER) == 1
    cpi.update_markdown(out, block.replace("**FAIL**", "**PASS**"))
    second = out.read_text(encoding="utf-8")
    assert second.count(cpi.START_MARKER) == 1 and "**PASS**" in second and "**FAIL**" not in second
    assert second.startswith("# Data\n\nintro\n")


def test_sample_is_deterministic_and_skips_missing_pdfs(cpi: ModuleType) -> None:
    questions = [_question(f"q{i:02d}", 1, "some evidence text long enough") for i in range(10)]
    questions.append(_question("zz", 1, "evidence", doc="MISSING_DOC"))
    a = cpi.sample_questions(questions, 4, 0, {DOC})
    b = cpi.sample_questions(questions, 4, 0, {DOC})
    assert [q.id for q in a] == [q.id for q in b] and len(a) == 4
    assert all(q.doc_name == DOC for q in a)
    assert [q.id for q in cpi.sample_questions(questions, 4, 1, {DOC})] != [q.id for q in a]


def test_main_rejects_bad_arguments(cpi: ModuleType) -> None:
    assert cpi.main(["--n", "0"]) == 2
    assert cpi.main(["--min-pass", "1.5"]) == 2
