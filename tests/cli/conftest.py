"""Fixtures for the CLI tests: a ``CliRunner`` helper, a synthetic FinanceBench-shaped corpus
(reportlab PDF + ``fb_mini.jsonl`` + a two-company ``companies.yaml``) and an index built
through ``secqa ingest financebench`` itself. Everything is offline; HTTP is respx-mocked.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from secqa.cli import app
from secqa.core.logging import configure_logging

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
FB_MINI = FIXTURES / "fb_mini.jsonl"
FIXTURE_PAGES = FIXTURES / "eval_fixture_pages.json"
GOLDEN_REPORT = FIXTURES / "eval_report_golden.md"
TOP_DOC = "FIXTURE_2023_10K"
EMBEDDER = "hashing:64"
TEST_UA = "Test Runner test@example.com"
NET_SALES_QUESTION = "What were total net sales in fiscal 2023?"

COMPANIES_YAML = """\
companies:
  - name: Fixture Corp
    ticker: FIXT
    cik: "0001234567"
    financebench_aliases: ["FIXTURE"]
  - name: Acme Holdings
    ticker: ACME
    cik: "0007654321"
    financebench_aliases: ["ACME"]
"""


@dataclass(frozen=True)
class Invoked:
    """A finished CLI invocation."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


Invoke = Callable[..., Invoked]


@pytest.fixture(autouse=True)
def _rebind_logging() -> Iterator[None]:
    """The CLI binds its log handler to the runner's stderr; rebind after each test."""
    yield
    configure_logging(json=False)
    sys.stderr.flush()


@pytest.fixture
def invoke() -> Invoke:
    """``invoke("doctor", "--offline")`` runs the CLI in-process; bugs raise, exits are returned."""
    runner = CliRunner()

    def _run(*args: str) -> Invoked:
        result = runner.invoke(app, list(args), catch_exceptions=False)
        return Invoked(result.exit_code, result.stdout, result.stderr)

    return _run


@pytest.fixture
def companies_path(tmp_path: Path) -> Path:
    path = tmp_path / "companies.yaml"
    path.write_text(COMPANIES_YAML, encoding="utf-8")
    return path


@pytest.fixture
def pdf_dir(tmp_path: Path, fixture_pdf_factory: Callable[..., Path]) -> Path:
    """``<tmp>/pdfs/FIXTURE_2023_10K.pdf`` drawn from the eval fixture pages."""
    import json

    docs = json.loads(FIXTURE_PAGES.read_text(encoding="utf-8"))
    directory = tmp_path / "pdfs"
    directory.mkdir()
    source = fixture_pdf_factory(pages=list(docs[TOP_DOC]["pages"]), name=TOP_DOC)
    source.replace(directory / f"{TOP_DOC}.pdf")
    return directory


@pytest.fixture
def index_path(tmp_path: Path) -> Path:
    return tmp_path / "index.duckdb"


@pytest.fixture
def built_index(invoke: Invoke, pdf_dir: Path, companies_path: Path, index_path: Path) -> Path:
    """An index built by ``secqa ingest financebench`` over the synthetic corpus."""
    result = invoke(
        "-q",
        "ingest",
        "financebench",
        "--pdf-dir",
        str(pdf_dir),
        "--questions",
        str(FB_MINI),
        "--companies",
        str(companies_path),
        "--embedder",
        EMBEDDER,
        "--db",
        str(index_path),
    )
    assert result.exit_code == 0, result.output
    return index_path
